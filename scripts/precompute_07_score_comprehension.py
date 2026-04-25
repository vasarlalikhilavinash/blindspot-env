#!/usr/bin/env python3
"""Stage 07 — Score onboarding comprehension per (user, concept).

Methodology (LLM-as-Judge, Prometheus-2 / Zheng et al. 2023 style):

    1. For each in-pool concept, generate 3 multiple-choice questions
       (4 options + correct letter) GROUNDED ON THE FULL CONCEPT ABSTRACT
       using a strong generator model (GPT-5.4). Cached per concept.

    2. For each (user, concept) where the concept is in the user's
       relevant pool (positives + hard_negatives), present TWO conditions
       to TWO independent judges (Gemini 3.1 Pro + GPT-5.4):

          A. CONTROL   — only the concept's one-line definition is shown
          B. TREATMENT — the full 5-paper reading-path summary is shown

       Each judge picks A/B/C/D for all 3 questions in a single request.
       Score per (judge, condition) ∈ {0/3, 1/3, 2/3, 3/3}.

    3. Comprehension = mean over judges of (treatment_acc − control_acc),
       clipped to [0, 1].

    4. Inter-judge agreement: Cohen's κ over the per-question correctness
       vector (length 6 = 3 questions × 2 conditions). If κ < 0.7 the
       signal is treated as noisy and the pair is dropped from the
       output (Zheng et al. recommend κ ≥ 0.6; we use 0.7 to be strict).

Outputs:
    scripts/_cache/comprehension_scores.json   {user_id: {concept_id_str: float}}
    scripts/_cache/judge_kappa.json            {concept_id_str: kappa}
    scripts/_cache/qa_bank.json                {concept_id_str: [{q,opts,answer}]}
    scripts/_cache/judge_responses.json        full responses, for audit

Models (April 2026, cost-optimized defaults):
    Generator : gpt-5.4-mini      ($0.75/M in, $4.50/M out)
    Judge A   : gpt-5.4-mini      (cheap, intel=49)
    Judge B   : gemini-2.5-flash    ($0.30/M in, $2.50/M out, intel=46)

    Override via env if you want frontier judges:
        BLINDSPOT_JUDGE_OPENAI=gpt-5.4
        BLINDSPOT_JUDGE_GEMINI=gemini-3.1-pro-preview
        BLINDSPOT_GEN_MODEL=gpt-5.4

Required env vars:
    OPENAI_API_KEY      (https://platform.openai.com/api-keys)
    GEMINI_API_KEY      (https://aistudio.google.com/apikey)

Optional:
    BLINDSPOT_MAX_USERS=N       cap users for smoke-test
    BLINDSPOT_MAX_CONCEPTS=N    cap concepts per user (positives+HN only)
    BLINDSPOT_DRY_RUN=1         skip API calls; emit zeros (for CI)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

CACHE_DIR = Path(__file__).resolve().parent / "_cache"
STRUCT_IN = CACHE_DIR / "pool_structured.json"
PATHS_IN = CACHE_DIR / "reading_paths.json"
CONCEPTS_IN = CACHE_DIR / "concepts.json"

SCORES_OUT = CACHE_DIR / "comprehension_scores.json"
KAPPA_OUT = CACHE_DIR / "judge_kappa.json"
QA_BANK = CACHE_DIR / "qa_bank.json"
RESP_LOG = CACHE_DIR / "judge_responses.json"

KAPPA_THRESHOLD = 0.7
N_QUESTIONS = 3
GEN_MODEL = os.environ.get("BLINDSPOT_GEN_MODEL", "gpt-5.4-mini")
JUDGE_OPENAI = os.environ.get("BLINDSPOT_JUDGE_OPENAI", "gpt-5.4-mini")
JUDGE_GEMINI = os.environ.get("BLINDSPOT_JUDGE_GEMINI", "gemini-2.5-flash")

DRY_RUN = os.environ.get("BLINDSPOT_DRY_RUN") == "1"
MAX_USERS = int(os.environ.get("BLINDSPOT_MAX_USERS", "0")) or None
# Default cap of 10 concepts per user (5 positives + 5 hard_negatives) keeps
# total API cost under ~$6 with mini/flash judges. Override or unset for full run.
MAX_CONCEPTS = int(os.environ.get("BLINDSPOT_MAX_CONCEPTS", "10")) or None


# ─────────────────────────── SDK initialization ───────────────────────────

_openai_client = None
_gemini_client = None


def openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI  # type: ignore
        _openai_client = OpenAI()  # picks up OPENAI_API_KEY
    return _openai_client


def gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # type: ignore
        _gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _gemini_client


# ───────────────────────────── helpers ──────────────────────────────

def cohen_kappa(a: List[int], b: List[int]) -> float:
    """Binary Cohen's kappa for matched-length 0/1 lists."""
    n = len(a)
    if n == 0:
        return 0.0
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if agree == 1.0 else 0.0
    return (agree - pe) / (1 - pe)


def _strict_json(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
    return None


def _retry(fn, *, attempts: int = 4, base: float = 1.5):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(base ** i)
    raise RuntimeError(f"giving up after {attempts} attempts: {last}")


# ───────────────────────────── LLM calls ─────────────────────────────

def call_openai(model: str, system: str, user: str, *, max_tokens: int = 600) -> str:
    if DRY_RUN:
        return ""
    def _go():
        rsp = openai_client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return rsp.choices[0].message.content or ""
    return _retry(_go)


def call_gemini(model: str, system: str, user: str, *, max_tokens: int = 600) -> str:
    if DRY_RUN:
        return ""
    def _go():
        from google.genai import types  # type: ignore
        rsp = gemini_client().models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                # Disable thinking on 2.5/3.x models — saves output tokens and is
                # unnecessary for short MCQ answers grounded on provided context.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return rsp.text or ""
    return _retry(_go)


# ───────────────────────────── prompts ─────────────────────────────

GEN_SYS = (
    "You write factual multiple-choice questions for graduate-level ML/AI "
    "researchers. Output STRICT JSON only."
)


def gen_questions(concept: dict, path: List[dict] | None = None) -> List[dict]:
    abstract = concept.get("abstract_summary") or concept.get("one_liner", "")
    one_liner = concept.get("one_liner", "")
    title = concept["title"]
    path = path or []
    path_block = ""
    if path:
        lines = []
        for i, p in enumerate(path, 1):
            lines.append(f"  {i}. {p.get('title','?')} ({p.get('year','?')})")
        path_block = "\nReading path (foundational \u2192 recent):\n" + "\n".join(lines)

    user = (
        f"Concept title: {title}\n"
        f"One-line definition (this is what a SHALLOW reader knows):\n  {one_liner}\n\n"
        f"Full abstract / extended definition:\n{abstract}\n"
        f"{path_block}\n\n"
        f"Write exactly {N_QUESTIONS} multiple-choice questions (4 options each, "
        f"one correct). HARD CONSTRAINTS:\n"
        f"  1. Each question MUST require a specific fact from the abstract or "
        f"reading path \u2014 NOT something derivable from the one-line definition or "
        f"the title alone.\n"
        f"  2. Probe mechanism, named techniques, evaluation findings, ablation "
        f"results, or paper-specific contributions. Avoid surface-level 'what is X' "
        f"questions.\n"
        f"  3. Distractors must be plausible to a reader who only saw the one-liner "
        f"\u2014 i.e. they should be on-topic for the field but factually wrong for "
        f"THIS concept's body of work.\n"
        f"  4. Do NOT include the answer or any clear paraphrase of the one-liner in "
        f"the question stem.\n\n"
        f'Return JSON: {{"questions":[{{"q":"...","options":["A...","B...","C...","D..."],"answer":"A"}}]}}'
    )
    raw = call_openai(GEN_MODEL, GEN_SYS, user, max_tokens=1200)
    obj = _strict_json(raw) or {}
    qs = obj.get("questions") if isinstance(obj, dict) else None
    if not qs or len(qs) < N_QUESTIONS:
        return []
    cleaned = []
    for q in qs[:N_QUESTIONS]:
        if not all(k in q for k in ("q", "options", "answer")):
            continue
        if q["answer"] not in ("A", "B", "C", "D") or len(q["options"]) != 4:
            continue
        # Store the correct option's TEXT (not its position) so we can
        # shuffle the option order per-judge-call to eliminate positional bias.
        correct_idx = ord(q["answer"]) - ord("A")
        cleaned.append({
            "q": q["q"],
            "options": q["options"],
            "correct_text": q["options"][correct_idx],
        })
    return cleaned if len(cleaned) == N_QUESTIONS else []


JUDGE_SYS = (
    "You are answering multiple-choice questions about a research concept. "
    "Use ONLY the context provided. If unsure, pick the best supported option. "
    "Output STRICT JSON only."
)


def render_questions(qs: List[dict]) -> str:
    out = []
    for i, q in enumerate(qs, 1):
        opts = "\n".join(f"  {chr(64+j+1)}. {o}" for j, o in enumerate(q["options"]))
        out.append(f"Q{i}. {q['q']}\n{opts}")
    return "\n\n".join(out)


def shuffle_qs(qs: List[dict], seed: int) -> tuple[List[dict], List[str]]:
    """Shuffle options per question deterministically; return shuffled qs + new gold letters."""
    import random
    rng = random.Random(seed)
    new_qs: List[dict] = []
    gold_letters: List[str] = []
    for q in qs:
        opts = list(q["options"])
        rng.shuffle(opts)
        try:
            gold_idx = opts.index(q["correct_text"])
        except ValueError:
            # Backward-compat for old QA bank that stored "answer" letter only
            gold_idx = ord(q.get("answer", "A")) - ord("A")
        new_qs.append({"q": q["q"], "options": opts})
        gold_letters.append(chr(ord("A") + gold_idx))
    return new_qs, gold_letters


def judge_answer(judge: str, context: str, qs: List[dict]) -> List[str]:
    user = (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTIONS:\n{render_questions(qs)}\n\n"
        f'Return JSON: {{"answers":["A","B","C"]}} (one letter per question, in order).'
    )
    if judge == "openai":
        raw = call_openai(JUDGE_OPENAI, JUDGE_SYS, user, max_tokens=120)
    elif judge == "gemini":
        raw = call_gemini(JUDGE_GEMINI, JUDGE_SYS, user, max_tokens=120)
    else:
        raise ValueError(judge)
    obj = _strict_json(raw) or {}
    ans = obj.get("answers") if isinstance(obj, dict) else None
    if not isinstance(ans, list):
        return [""] * N_QUESTIONS
    out = []
    for a in ans[:N_QUESTIONS]:
        a = str(a).strip().upper()[:1]
        out.append(a if a in "ABCD" else "")
    while len(out) < N_QUESTIONS:
        out.append("")
    return out


# ───────────────────── reading-path → context string ─────────────────

def path_context(concept: dict, path: List[dict]) -> str:
    head = f"Concept: {concept['title']}\nDefinition: {concept.get('abstract_summary') or concept.get('one_liner','')}\n"
    if not path:
        return head + "\n(No reading path available — relying on definition only.)"
    head += "\nReading path (foundational → recent):\n"
    for i, p in enumerate(path, 1):
        head += f"  {i}. {p.get('title','?')} ({p.get('year','?')})\n"
    return head


def control_context(concept: dict) -> str:
    return (
        f"Concept: {concept['title']}\n"
        f"Definition: {concept.get('one_liner','')}\n"
    )


# ───────────────────────────── main ─────────────────────────────

def main():  # noqa: C901
    if not DRY_RUN:
        if "OPENAI_API_KEY" not in os.environ:
            sys.exit("ERROR: OPENAI_API_KEY not set (or run with BLINDSPOT_DRY_RUN=1)")
        if "GEMINI_API_KEY" not in os.environ:
            sys.exit("ERROR: GEMINI_API_KEY not set (or run with BLINDSPOT_DRY_RUN=1)")

    structured = json.loads(STRUCT_IN.read_text())
    paths = json.loads(PATHS_IN.read_text()) if PATHS_IN.exists() else {}
    concepts_list = json.loads(CONCEPTS_IN.read_text())
    concepts = {c["concept_id"]: c for c in concepts_list}

    qa_bank: Dict[str, List[dict]] = (
        json.loads(QA_BANK.read_text()) if QA_BANK.exists() else {}
    )
    resp_log: Dict[str, dict] = (
        json.loads(RESP_LOG.read_text()) if RESP_LOG.exists() else {}
    )
    scores: Dict[str, Dict[str, float]] = (
        json.loads(SCORES_OUT.read_text()) if SCORES_OUT.exists() else {}
    )
    kappa_record: Dict[str, float] = (
        json.loads(KAPPA_OUT.read_text()) if KAPPA_OUT.exists() else {}
    )

    user_items = list(structured.items())
    if MAX_USERS:
        user_items = user_items[:MAX_USERS]

    total_pairs = sum(
        min(MAX_CONCEPTS or 10**9, len(p.get("positives", []) + p.get("hard_negatives", [])))
        for _, p in user_items
    )
    done = 0
    print(f"Stage 07: scoring ~{total_pairs} (user, concept) pairs "
          f"with {JUDGE_OPENAI} + {JUDGE_GEMINI}")

    for uid, p in user_items:
        scores.setdefault(uid, {})
        # Interleave positives and hard_negatives so MAX_CONCEPTS keeps both classes
        pos = list(p.get("positives", []))
        neg = list(p.get("hard_negatives", []))
        targets: List[int] = []
        for a, b in zip(pos, neg):
            targets.extend([a, b])
        targets.extend(pos[len(neg):])
        targets.extend(neg[len(pos):])
        if MAX_CONCEPTS:
            targets = targets[:MAX_CONCEPTS]

        for cid in targets:
            done += 1
            cs = str(cid)
            if cs in scores[uid]:
                continue  # resume

            concept = concepts.get(cid)
            if not concept:
                continue

            # 1) Question bank (per concept, cached)
            qs = qa_bank.get(cs)
            if not qs:
                qs = gen_questions(concept, paths.get(cs, []))
                if not qs:
                    print(f"  [skip] {cs} ({concept['title'][:40]}) — QA gen failed")
                    continue
                qa_bank[cs] = qs
                QA_BANK.write_text(json.dumps(qa_bank, indent=2))

            # 2) Run judges in both conditions, with options shuffled
            #    independently per (judge, condition) to eliminate positional
            #    bias. Different shuffles → judges score on the same questions
            #    but never on identical (option_order, gold_letter) pairs.
            ctx_ctrl = control_context(concept)
            ctx_treat = path_context(concept, paths.get(cs, []))
            try:
                base_seed = (hash(uid) ^ cid) & 0xFFFFFFFF
                qs_oc, gold_oc = shuffle_qs(qs, base_seed + 1)
                qs_gc, gold_gc = shuffle_qs(qs, base_seed + 2)
                qs_ot, gold_ot = shuffle_qs(qs, base_seed + 3)
                qs_gt, gold_gt = shuffle_qs(qs, base_seed + 4)
                a_o_ctrl = judge_answer("openai", ctx_ctrl, qs_oc)
                a_g_ctrl = judge_answer("gemini", ctx_ctrl, qs_gc)
                a_o_treat = judge_answer("openai", ctx_treat, qs_ot)
                a_g_treat = judge_answer("gemini", ctx_treat, qs_gt)
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] {cs}: judge error {e}")
                continue

            # 3) Score correctness — each judge/condition has its own gold
            #    letter sequence (since options were shuffled differently).
            corr_o_ctrl = [int(a == g) for a, g in zip(a_o_ctrl, gold_oc)]
            corr_g_ctrl = [int(a == g) for a, g in zip(a_g_ctrl, gold_gc)]
            corr_o_treat = [int(a == g) for a, g in zip(a_o_treat, gold_ot)]
            corr_g_treat = [int(a == g) for a, g in zip(a_g_treat, gold_gt)]

            # 4) Inter-judge κ over the full 6-vector (3 questions × 2 conditions)
            kappa = cohen_kappa(
                corr_o_ctrl + corr_o_treat,
                corr_g_ctrl + corr_g_treat,
            )
            kappa_record[cs] = round(kappa, 3)

            resp_log[f"{uid}:{cs}"] = {
                "title": concept["title"],
                "answers": {
                    "openai_control": a_o_ctrl, "openai_treatment": a_o_treat,
                    "gemini_control": a_g_ctrl, "gemini_treatment": a_g_treat,
                },
                "gold": {
                    "openai_control": gold_oc, "openai_treatment": gold_ot,
                    "gemini_control": gold_gc, "gemini_treatment": gold_gt,
                },
                "kappa": round(kappa, 3),
            }

            if kappa < KAPPA_THRESHOLD:
                # judges disagree — drop signal
                pass
            else:
                lift_o = mean(corr_o_treat) - mean(corr_o_ctrl)
                lift_g = mean(corr_g_treat) - mean(corr_g_ctrl)
                lift = max(0.0, min(1.0, (lift_o + lift_g) / 2))
                if lift > 0:
                    scores[uid][cs] = round(lift, 3)

            # checkpoint every 25 pairs
            if done % 25 == 0:
                SCORES_OUT.write_text(json.dumps(scores, indent=2))
                KAPPA_OUT.write_text(json.dumps(kappa_record, indent=2))
                RESP_LOG.write_text(json.dumps(resp_log, indent=2))
                print(f"  {done}/{total_pairs} pairs done "
                      f"(latest κ={kappa:.2f}, scored={sum(len(v) for v in scores.values())})")

    SCORES_OUT.write_text(json.dumps(scores, indent=2))
    KAPPA_OUT.write_text(json.dumps(kappa_record, indent=2))
    RESP_LOG.write_text(json.dumps(resp_log, indent=2))

    n_scored = sum(len(v) for v in scores.values())
    mean_k = mean(kappa_record.values()) if kappa_record else 0.0
    print(f"Done. Scored {n_scored} (user, concept) pairs across {len(scores)} users.")
    print(f"Mean inter-judge κ = {mean_k:.3f} (threshold {KAPPA_THRESHOLD}).")


if __name__ == "__main__":
    main()
