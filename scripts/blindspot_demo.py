#!/usr/bin/env python3
"""Blindspot judge-facing demo.

Pipeline:
    paragraph (LinkedIn bio / job description)
        ↓ TF-IDF profile build (on-the-fly, no precompute)
        ↓ closest-of-17-users match (cosine over user summaries)
        ↓ candidate pool = top-50 most-relevant concepts by tfidf×concept-text
        ↓ run 5 policies: Random / Trending / Dense / Blindspot-RL / GPT-baseline
        ↓ for each surfaced concept: pull reading-path + comprehension-lift
        ↓ render side-by-side comparison + reward decomposition

Designed to be importable from notebook OR run as CLI.

CLI:
    python scripts/blindspot_demo.py "I'm a senior ML engineer at a fintech..."
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"


# ─────────────────────────── data loading ───────────────────────────

def _load() -> Dict[str, Any]:
    return {
        "users": json.load(open(DATA / "user_summaries.json")),
        "catalog": json.load(open(DATA / "concept_catalog.json")),
        "pool": json.load(open(DATA / "concept_pool_per_user.json")),
        "adoption": json.load(open(DATA / "ground_truth_adoption.json")),
        "comp": json.load(open(DATA / "comprehension_scores.json")),
        "paths": json.load(open(DATA / "reading_paths.json")),
        "novelty": json.load(open(DATA / "novelty_flags.json")),
        "knn": json.load(open(DATA / "knn_users.json")),
    }


# ─────────────────────────── tfidf-lite ───────────────────────────

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z\-_]{2,}")

STOPWORDS = set("""
a an and are as at be by for from has have he her his i in is it its of on or our
that the their them they this to was we were what when where which who will with you your
yours us my me but not no so if then than also can could should would may might must just
over under into out about more most less some any all each every other another such same
being been do does did done make made get got use used using using used new old recent
work working works worked field area thing things stuff way ways system systems team teams
company companies people user users problem problems solution solutions help helps
helped want wants wanted need needs needed know knew known like likes liked good best
currently lately recently still already even very much many one two three first last next
research researcher recent paper papers titles title across mentioned corpus
""".split())


def tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(s or "")
            if w.lower() not in STOPWORDS and len(w) > 2]


@dataclass
class Vocab:
    word2idx: Dict[str, int] = field(default_factory=dict)
    idf: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @classmethod
    def fit(cls, docs: List[str]) -> "Vocab":
        df: Dict[str, int] = {}
        toks = [tokenize(d) for d in docs]
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        # keep top 8k by df
        keep = sorted(df.items(), key=lambda kv: -kv[1])[:8000]
        word2idx = {w: i for i, (w, _) in enumerate(keep)}
        n = len(docs)
        idf = np.zeros(len(word2idx))
        for w, i in word2idx.items():
            idf[i] = np.log((1 + n) / (1 + df[w])) + 1.0
        return cls(word2idx, idf)

    def transform(self, doc: str) -> np.ndarray:
        v = np.zeros(len(self.word2idx))
        for w in tokenize(doc):
            j = self.word2idx.get(w)
            if j is not None:
                v[j] += 1.0
        v *= self.idf
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


# ─────────────────────────── core demo engine ───────────────────────────

class BlindspotDemo:
    """Stateless engine — build once, query many times."""

    def __init__(self, llm_generate: Optional[Callable[[str], str]] = None,
                 openai_compare: Optional[Callable[[str], str]] = None):
        self.d = _load()
        self.users = self.d["users"]
        self.catalog = self.d["catalog"]
        self.user_ids = list(self.users.keys())
        self.concept_ids = list(self.catalog.keys())

        # Build TF-IDF over (users + concept descriptions) so query/doc share vocab
        user_docs = [self.users[u] for u in self.user_ids]
        concept_docs = [self._concept_text(c) for c in self.concept_ids]
        self.vocab = Vocab.fit(user_docs + concept_docs)
        self.user_vecs = np.stack([self.vocab.transform(d) for d in user_docs])
        self.concept_vecs = np.stack([self.vocab.transform(d) for d in concept_docs])

        # Optional callables for the trained policy + a generic GPT-4 baseline
        self.llm_generate = llm_generate     # (prompt) -> single completion (action JSON)
        self.openai_compare = openai_compare  # (paragraph) -> generic advice text

        # Optional pre-computed cache: key -> list of surfaced concept_ids for the
        # trained policy. Keys are either user_id (str) or 'persona::<name>'.
        # Generated by scripts/precompute_demo_cache.py and loaded if present —
        # lets the hosted Space serve trained-policy results with ZERO GPU.
        cache_path = DATA / "demo_cache.json"
        try:
            self.demo_cache = json.load(open(cache_path)) if cache_path.exists() else {}
        except Exception:
            self.demo_cache = {}

    # ------------------------------- helpers -------------------------------

    def _concept_text(self, cid: str) -> str:
        c = self.catalog.get(str(cid), {})
        return f"{c.get('title','')}. {c.get('one_liner','')}. {c.get('abstract_summary','')}"

    def _adopted_for_user(self, uid: str) -> set:
        return set(str(c) for c in (self.d["adoption"].get(uid) or {}).keys())

    def _knn_adopted(self, uid: str) -> set:
        out = set()
        for nid in self.d["knn"].get(uid, []):
            out |= set(str(c) for c in (self.d["adoption"].get(nid) or {}).keys())
        return out

    # ------------------------------- profile build -------------------------------

    def build_profile(self, paragraph: str) -> Dict[str, Any]:
        """TF-IDF the paragraph, find the closest of our 17 users, and explain why."""
        q = self.vocab.transform(paragraph)
        sims = self.user_vecs @ q
        order = np.argsort(-sims)
        top_uid = self.user_ids[int(order[0])]
        top_sim = float(sims[order[0]])

        # Which keywords drove the match?
        idx2word = {i: w for w, i in self.vocab.word2idx.items()}
        contrib = q * self.user_vecs[order[0]]
        top_words = [idx2word[int(i)] for i in np.argsort(-contrib)[:8] if contrib[int(i)] > 0]

        return {
            "matched_user_id": top_uid,
            "match_similarity": top_sim,
            "matched_summary": self.users[top_uid][:300],
            "shared_keywords": top_words,
            "query_vec": q,
        }

    def build_candidates(self, profile: Dict[str, Any], k: int = 50) -> List[str]:
        """50 candidate concepts: 30 most-relevant + 10 trending bait + 10 random noise."""
        q = profile["query_vec"]
        sims = self.concept_vecs @ q
        ranked = list(np.argsort(-sims))
        relevant = [self.concept_ids[int(i)] for i in ranked[:30]]
        trending = [c for c in self.concept_ids
                    if self.catalog[c].get("is_trending") and c not in relevant][:10]
        rng = random.Random(0)
        noise_pool = [c for c in self.concept_ids if c not in relevant and c not in trending]
        noise = rng.sample(noise_pool, k=min(10, len(noise_pool)))
        out = relevant + trending + noise
        rng.shuffle(out)
        return out[:k]

    # ------------------------------- reward (mirrors server/rewards.py) -------------------------------

    def _reward_for(self, uid: str, surfaced: List[str]) -> Dict[str, float]:
        adoption = self.d["adoption"].get(uid, {})
        knn_adopt = self._knn_adopted(uid)
        novelty = self.d["novelty"]
        comp = self.d["comp"].get(uid, {})

        r = {"adoption": 0.0, "novelty": 0.0, "onboarding": 0.0,
             "false_positive": 0.0, "efficiency": 0.0, "total": 0.0}
        for cid in surfaced:
            cid_s = str(cid)
            score = float(adoption.get(cid_s, 0.0))
            adopted = score >= 1e-6
            if not adopted and cid_s in knn_adopt:
                score = 0.3
                adopted = True
            r["adoption"] += score
            if adopted:
                r["novelty"] += 0.5 * (1.0 if novelty.get(cid_s) else 0.0)
                r["onboarding"] += float(comp.get(cid_s, 0.0))
            else:
                r["false_positive"] -= 0.1
        r["total"] = sum(r[k] for k in
                         ("adoption", "novelty", "onboarding", "false_positive", "efficiency"))
        return r

    # ------------------------------- policies -------------------------------

    def policy_random(self, profile, candidates, k=3) -> Tuple[List[str], str]:
        rng = random.Random(0)
        return rng.sample(candidates, k), "Picks 3 concepts uniformly at random."

    def policy_trending(self, profile, candidates, k=3) -> Tuple[List[str], str]:
        ranked = sorted(candidates,
                        key=lambda c: -self.catalog[c].get("growth_signal", 0))
        return ranked[:k], "Picks the 3 most-mentioned concepts (popularity bait)."

    def policy_dense_retrieval(self, profile, candidates, k=3) -> Tuple[List[str], str]:
        q = profile["query_vec"]
        sims = [(c, float(self.concept_vecs[self.concept_ids.index(c)] @ q))
                for c in candidates]
        sims.sort(key=lambda x: -x[1])
        return [c for c, _ in sims[:k]], "Picks the 3 cosine-nearest concepts (you likely already know these)."

    def policy_blindspot(self, profile, candidates, k=3,
                         cache_key: Optional[str] = None) -> Tuple[List[str], str, Dict[str, Any]]:
        """Trained RL agent. Resolution order:
          1. Pre-computed cache (instant, no GPU needed)  -- ideal for hosted Space
          2. Live `self.llm_generate` callable (when adapter is loaded in-process)
          3. kNN-informed proxy (pre-training placeholder)
        """
        meta = {"used_trained_model": False, "used_cache": False, "reasoning": ""}

        # 1. Cache hit
        if cache_key and cache_key in self.demo_cache:
            entry = self.demo_cache[cache_key]
            picked = [str(c) for c in entry.get("surfaced", [])][:k]
            if picked:
                meta["used_trained_model"] = True
                meta["used_cache"] = True
                meta["reasoning"] = entry.get("reasoning", "(served from precomputed trained-policy cache)")
                return picked, "Trained Blindspot policy (GRPO-finetuned Qwen-9B, served from cache).", meta

        # 2. Live model
        if self.llm_generate is not None:
            try:
                surfaced, reasoning = self._run_trained_loop(profile, candidates, k)
                meta["used_trained_model"] = True
                meta["reasoning"] = reasoning
                return surfaced, "Trained Blindspot policy (GRPO-finetuned Qwen-9B, live).", meta
            except Exception as e:
                meta["reasoning"] = f"(trained model failed: {e}; using analytical proxy)"

        # 3. Pre-training placeholder: kNN-informed proxy.
        uid = profile["matched_user_id"]
        knn_adopted = self._knn_adopted(uid)
        q = profile["query_vec"]
        rel_rank = {c: i for i, c in enumerate(
            sorted(candidates,
                   key=lambda c: -float(self.concept_vecs[self.concept_ids.index(c)] @ q)))}
        scored = []
        for c in candidates:
            knn_signal = 1.0 if str(c) in knn_adopted else 0.0
            novel = 1.0 if not self.catalog[c].get("is_trending") else 0.4
            rel = 1.0 / (1 + rel_rank[c])
            scored.append((c, knn_signal * 2.0 + novel + rel))
        scored.sort(key=lambda x: -x[1])
        picked = [s[0] for s in scored[:k]]
        meta["reasoning"] = (
            "⚠️ Trained model not loaded — using kNN-informed proxy as placeholder.\n"
            "Real trained policy will be plugged in once GRPO finishes.\n"
            "Proxy ranking:  2·kNN-adopted-signal + novelty-bonus + relevance."
        )
        return picked, "Blindspot proxy (pre-training placeholder).", meta

    def _run_trained_loop(self, profile, candidates, k):
        """Drive the trained LLM through a quick episode."""
        from training.grpo_train import SYSTEM_PROMPT
        cand_lines = "\n".join(
            f"  id={c}: {self.catalog[c].get('title','')[:90]}" for c in candidates
        )
        obs_text = (
            f"USER:\n{profile['matched_summary']}\n\n"
            f"BUDGETS i=0 s={k}\n\nCANDIDATES:\n{cand_lines}"
        )
        surfaced: List[str] = []
        reasoning_parts: List[str] = []
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]
        for _ in range(k * 2):
            text = self.llm_generate(msgs)
            reasoning_parts.append(text.strip()[:200])
            try:
                m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
                action = json.loads(m.group(0)) if m else {}
            except Exception:
                action = {}
            if action.get("type") == "surface":
                cid = str(action.get("concept_id"))
                if cid in candidates and cid not in surfaced:
                    surfaced.append(cid)
                    if len(surfaced) >= k:
                        break
            elif action.get("type") == "stop":
                break
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content": "Surface another, or stop."})
        return surfaced[:k], "\n".join(reasoning_parts)

    # ------------------------------- per-concept render -------------------------------

    def render_concept(self, cid: str, uid: str) -> Dict[str, Any]:
        cid_s = str(cid)
        cat = self.catalog.get(cid_s, {})
        adoption_score = float(self.d["adoption"].get(uid, {}).get(cid_s, 0.0))
        adopted = adoption_score >= 1e-6
        if not adopted and cid_s in self._knn_adopted(uid):
            adoption_score = 0.3
            adopted = True
        return {
            "concept_id": cid_s,
            "title": cat.get("title", "?"),
            "one_liner": cat.get("one_liner", ""),
            "is_trending": bool(cat.get("is_trending")),
            "growth_signal": float(cat.get("growth_signal", 0.0)),
            "is_novel": not bool(cat.get("is_trending")),
            "adopted_by_user": adopted,
            "adoption_score": adoption_score,
            "comprehension_lift": float(self.d["comp"].get(uid, {}).get(cid_s, 0.0)),
            "reading_path": self.d["paths"].get(cid_s, [])[:5],
            "verdict": self._verdict(adopted, cat),
        }

    def _verdict(self, adopted: bool, cat: Dict[str, Any]) -> str:
        if adopted and not cat.get("is_trending"):
            return "✅ HIT — true unknown-unknown that the user adopted later"
        if adopted and cat.get("is_trending"):
            return "✅ hit, but trending (lower novelty bonus)"
        if cat.get("is_trending"):
            return "❌ MISS — trending bait, no real adoption signal"
        return "❌ MISS — relevant on the surface but user did not adopt"

    # ------------------------------- top-level compare -------------------------------

    def compare_all(self, paragraph: Optional[str] = None,
                    user_id: Optional[str] = None,
                    persona_key: Optional[str] = None) -> Dict[str, Any]:
        """Three modes:
          • paragraph=...     — TF-IDF match to closest of 17 users (ad-hoc)
          • user_id=...       — directly use one of the 17 real users (cache key=user_id)
          • persona_key=...   — pre-canned persona (cache key='persona::<key>')
        """
        if user_id is not None:
            assert user_id in self.users, f"unknown user_id {user_id}"
            q = self.vocab.transform(self.users[user_id])
            profile = {
                "matched_user_id": user_id,
                "match_similarity": 1.0,
                "matched_summary": self.users[user_id][:300],
                "shared_keywords": [],
                "query_vec": q,
                "mode": "real-user",
            }
            # Use the user's pre-computed pool (real ground truth applies)
            candidates = [str(c) for c in self.d["pool"].get(user_id, [])][:50]
            if not candidates:
                candidates = self.build_candidates(profile)
            cache_key = f"user::{user_id}"
        elif persona_key is not None:
            # Persona text passed via paragraph; cache key is the persona name
            profile = self.build_profile(paragraph or "")
            profile["mode"] = "persona"
            candidates = self.build_candidates(profile)
            cache_key = f"persona::{persona_key}"
        else:
            profile = self.build_profile(paragraph or "")
            profile["mode"] = "paragraph-match"
            candidates = self.build_candidates(profile)
            cache_key = None  # ad-hoc paragraph: no cache, must use live model or proxy
        uid = profile["matched_user_id"]

        out: Dict[str, Any] = {
            "profile": {
                "matched_user_id": uid,
                "match_similarity": profile["match_similarity"],
                "matched_summary": profile["matched_summary"],
                "shared_keywords": profile.get("shared_keywords", []),
                "candidate_pool_size": len(candidates),
                "mode": profile.get("mode", "paragraph-match"),
                "weak_match_warning": (
                    profile["match_similarity"] < 0.25
                    and profile.get("mode") == "paragraph-match"
                ),
            },
            "policies": {},
        }

        runners = [
            ("Random",   self.policy_random),
            ("Trending", self.policy_trending),
            ("Dense Retrieval", self.policy_dense_retrieval),
            ("Blindspot RL",   self.policy_blindspot),
        ]
        for name, fn in runners:
            if name == "Blindspot RL":
                surfaced, descr, meta = fn(profile, candidates, cache_key=cache_key)
            else:
                surfaced, descr = fn(profile, candidates)
                meta = {}
            cards = [self.render_concept(c, uid) for c in surfaced]
            r = self._reward_for(uid, surfaced)
            out["policies"][name] = {
                "description": descr,
                "surfaced": surfaced,
                "cards": cards,
                "reward": r,
                "meta": meta,
            }

        # GPT-4 generic comparison (best-effort)
        if self.openai_compare is not None:
            try:
                out["chatgpt_baseline"] = self.openai_compare(paragraph)
            except Exception as e:
                out["chatgpt_baseline"] = f"(GPT comparison failed: {e})"
        else:
            out["chatgpt_baseline"] = (
                "(GPT comparison disabled — set OPENAI_API_KEY to enable)\n\n"
                "Generic advice ChatGPT would give without your specific concept catalog: "
                "it would recommend popular trending topics like 'agents', 'multimodal', "
                "'reasoning' — useful but not unknown-unknowns specific to YOUR work."
            )
        return out


# ─────────────────────────── pretty CLI render ───────────────────────────

def _wrap(s: str, w: int = 78) -> str:
    return textwrap.fill(s, width=w, initial_indent="    ", subsequent_indent="    ")


def render_text(report: Dict[str, Any]) -> str:
    out = []
    p = report["profile"]
    out.append("=" * 80)
    out.append("BLINDSPOT — UNKNOWN-UNKNOWNS DISCOVERY DEMO")
    out.append("=" * 80)
    out.append(f"\nProfile match : user_id={p['matched_user_id']}  cosine={p['match_similarity']:.3f}")
    out.append(f"Shared keys   : {', '.join(p['shared_keywords'][:6])}")
    out.append(f"Candidate pool: {p['candidate_pool_size']} concepts (relevant + trending + noise)")
    out.append(f"Matched user  : {p['matched_summary'][:160]}…\n")

    for name, res in report["policies"].items():
        out.append("─" * 80)
        rwd = res["reward"]
        out.append(f"  {name:18s} →  reward {rwd['total']:+.3f}   "
                   f"(adopt {rwd['adoption']:+.2f} · novel {rwd['novelty']:+.2f} · "
                   f"onboard {rwd['onboarding']:+.2f} · FP {rwd['false_positive']:+.2f})")
        out.append(f"  {res['description']}")
        for card in res["cards"]:
            out.append(f"    • [{card['concept_id']}] {card['title'][:60]}")
            out.append(f"        {card['verdict']}")
            if card["reading_path"]:
                out.append(f"        reading path ({len(card['reading_path'])} papers, "
                           f"comp-lift {card['comprehension_lift']:+.2f}):")
                for paper in card["reading_path"][:3]:
                    out.append(f"          – {paper.get('title','?')[:70]} ({paper.get('year','?')})")

    out.append("─" * 80)
    out.append("\nWhat a generic LLM (without your concept catalog) would say:")
    out.append(_wrap(report["chatgpt_baseline"]))
    out.append("=" * 80)
    return "\n".join(out)


# ─────────────────────────── CLI entrypoint ───────────────────────────

if __name__ == "__main__":
    paragraph = " ".join(sys.argv[1:]) or (
        "I'm a senior ML engineer at a fintech, building retrieval-augmented "
        "LLM systems for compliance Q&A. I work with embeddings, vector search, "
        "and prompt engineering. I want to know what I'm missing."
    )
    demo = BlindspotDemo()
    print(render_text(demo.compare_all(paragraph)))
