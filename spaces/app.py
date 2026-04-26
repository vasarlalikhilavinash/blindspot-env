"""Hugging Face Space entrypoint for Blindspot demo.

Architecture:
  - No GPU. Trained-policy responses are served from data/demo_cache.json
    (precomputed in Colab and committed to the repo).
  - Pre-training responses served from data/demo_cache_pretrain.json so the
    Before/After toggle shows the actual lift from SFT training.
  - Baselines (Random/Trending/Dense) and the kNN proxy run on CPU instantly.
"""
from __future__ import annotations
from collections import Counter
import html
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.blindspot_demo import BlindspotDemo
from scripts.precompute_demo_cache import PERSONAS

import gradio as gr


# ──────────────────────────── optional remote LLM ────────────────────────────
HF_ENDPOINT = os.environ.get("HF_INFERENCE_ENDPOINT", "").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def remote_llm_generate(messages):
    if not HF_ENDPOINT or not HF_TOKEN:
        raise RuntimeError("no remote endpoint configured")
    import requests
    r = requests.post(
        f"{HF_ENDPOINT}/v1/chat/completions",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"model": "tgi", "messages": messages, "max_tokens": 64, "temperature": 0.0},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def gpt_compare(paragraph: str):
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "You are a knowledgeable advisor. Given a description of "
                           "someone's work, tell them in ONE short paragraph (max 80 "
                           "words) what important AI/ML concepts they should be tracking "
                           "but probably aren't."
            }, {"role": "user", "content": paragraph}],
            max_tokens=200, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(GPT comparison failed: {e})"


# ──────────────────────────── engine ────────────────────────────
demo_engine = BlindspotDemo(
    llm_generate=remote_llm_generate if (HF_ENDPOINT and HF_TOKEN) else None,
    openai_compare=gpt_compare,
)
USER_OPTIONS = [(f"{uid}: {demo_engine.users[uid][:80]}…", uid)
                for uid in demo_engine.users.keys()]

PALETTE = {'Random': '#888', 'Trending': '#cc7733', 'Dense Retrieval': '#3377cc',
           'Blindspot (pre-training)': '#aa4488', 'Blindspot RL': '#22aa66'}

TOPIC_BUCKET_KEYWORDS = [
    ("Agents & Tool Use", ["agent", "tool", "workflow", "planner", "planning", "browser", "multi-agent"]),
    ("Retrieval & Search", ["retrieval", "rag", "search", "query", "passage", "rerank", "index"]),
    ("Reasoning & Alignment", ["reasoning", "alignment", "reward", "preference", "judge", "reflection", "policy"]),
    ("Diffusion & Multimodal", ["diffusion", "video", "image", "vision", "multimodal", "audio", "speech"]),
    ("Training & Fine-tuning", ["pretrain", "fine-tun", "lora", "adapter", "distill", "instruction", "tuning"]),
    ("Inference & Systems", ["inference", "serving", "cache", "kv", "latency", "throughput", "quantization", "speculative"]),
    ("Evaluation & Benchmarks", ["eval", "benchmark", "metric", "judge", "leaderboard"]),
    ("Safety & Robustness", ["safety", "privacy", "robust", "adversarial", "hallucination", "unlearning", "fairness"]),
    ("Domain Applications", ["legal", "biomedical", "medical", "recommendation", "dialogue", "summarization", "translation"]),
]


def infer_topic_bucket(title: str, one_liner: str = "") -> str:
    text = f"{title} {one_liner}".lower()
    scores = Counter()
    for bucket, keywords in TOPIC_BUCKET_KEYWORDS:
        for keyword in keywords:
            if keyword in text:
                scores[bucket] += 1
    if scores:
        return scores.most_common(1)[0][0]
    if "large language" in text or "llm" in text:
        return "General LLMs"
    return "Other"


# ──────────────────────────── Human research loop ────────────────────────────
def render_human_research_loop(report):
    """Replay the same research session for base model vs RL policy.

    Both columns use the same candidate pool and the same distraction tabs.
    The only thing that changes is what each policy ultimately bookmarks.
    """
    try:
        from server.blindspot_environment import BlindspotEnvironment
        from models import BlindspotAction
    except Exception as exc:
        return (
            "<div style='margin:28px 0;padding:18px;background:#fff8e1;border-radius:12px;"
            "border:1px solid #f0d58c;'>"
            "<b>Human research loop unavailable.</b><br>"
            f"<span style='font-size:12px;color:#666;'>{html.escape(str(exc))}</span></div>"
        )

    uid = report['profile'].get('matched_user_id')
    pre_policy = report['policies'].get('Blindspot (pre-training)', {})
    rl_policy = report['policies'].get('Blindspot RL', {})
    if not uid or not pre_policy.get('surfaced') or not rl_policy.get('surfaced'):
        return ""

    env = BlindspotEnvironment()
    obs = env.reset(user_id=uid, seed=0)
    candidate_lookup = {card.concept_id: card for card in obs.candidate_concepts}
    pre_ids = [int(cid) for cid in pre_policy.get('surfaced', [])
               if str(cid).isdigit() and int(cid) in candidate_lookup]
    rl_ids = [int(cid) for cid in rl_policy.get('surfaced', [])
              if str(cid).isdigit() and int(cid) in candidate_lookup]
    if not pre_ids or not rl_ids:
        return ""

    catalog = getattr(env, "_data").concept_catalog
    adoption_table = getattr(env, "_data").adoption.get(uid, {})

    def _catalog_record(cid: int):
        return catalog.get(cid, catalog.get(str(cid), {}))

    def _adopted_score(cid: int) -> float:
        return float(adoption_table.get(cid, adoption_table.get(str(cid), 0.0)))

    candidate_records = []
    for cid, card in candidate_lookup.items():
        rec = _catalog_record(cid)
        title = rec.get("title", card.title)
        one_liner = rec.get("one_liner", card.one_liner)
        candidate_records.append({
            "concept_id": cid,
            "title": title,
            "one_liner": one_liner,
            "is_trending": bool(rec.get("is_trending")),
            "growth_signal": float(rec.get("growth_signal", 0.0)),
            "adopted_by_user": _adopted_score(cid) >= 1e-6,
            "topic": infer_topic_bucket(title, one_liner),
        })
    candidate_by_id = {record["concept_id"]: record for record in candidate_records}

    session_ids = set(pre_ids) | set(rl_ids)
    shared_skip_ids = []
    for record in candidate_records:
        cid = record["concept_id"]
        if cid in session_ids or record["adopted_by_user"]:
            continue
        shared_skip_ids.append(cid)
        if len(shared_skip_ids) >= 2:
            break

    if len(shared_skip_ids) < 2:
        for record in candidate_records:
            cid = record["concept_id"]
            if cid in session_ids or cid in shared_skip_ids:
                continue
            shared_skip_ids.append(cid)
            if len(shared_skip_ids) >= 2:
                break

    start_titles = [html.escape(card.title) for card in obs.candidate_concepts[:5]]
    total_trending = sum(record["is_trending"] for record in candidate_records)
    total_under_radar = len(candidate_records) - total_trending
    total_adopted = sum(record["adopted_by_user"] for record in candidate_records)
    topic_counts = Counter(record["topic"] for record in candidate_records)

    def _chip(text, bg, fg):
        return (f"<span style='display:inline-block;margin-right:6px;margin-top:4px;"
                f"padding:3px 8px;border-radius:999px;background:{bg};color:{fg};"
                f"font-size:11px;font-weight:600;'>{text}</span>")

    def _mini_topic_card(record, accent, tone, extra_label=""):
        if not record:
            return ""
        chips = [
            _chip(record["topic"], tone, accent),
            _chip("Trending" if record["is_trending"] else "Under the radar",
                  "#fff3e0" if record["is_trending"] else "#e8f5e9",
                  "#ef6c00" if record["is_trending"] else "#2e7d32"),
        ]
        if extra_label:
            chips.append(_chip(extra_label, "#f5f5f5", "#555"))
        return (
            f"<div style='margin:8px 0;padding:10px 12px;background:white;border-left:4px solid {accent};"
            f"border-radius:0 7px 7px 0;box-shadow:0 1px 2px rgba(0,0,0,0.04);'>"
            f"<div style='font-size:13px;font-weight:700;'>{html.escape(record['title'][:54])}</div>"
            f"<div style='font-size:12px;color:#666;margin-top:4px;'>{html.escape(record['one_liner'][:120])}</div>"
            f"<div style='margin-top:6px;'>{''.join(chips)}</div>"
            f"</div>"
        )

    def _build_plan(saved_ids):
        plan = []
        if shared_skip_ids:
            plan.append(("inspect", shared_skip_ids[0],
                         "Open a tempting result first, like opening a browser tab that looks relevant."))
            plan.append(("skip", shared_skip_ids[0],
                         "Close that tab and move on — interesting on the surface, but not worth a bookmark."))
        if saved_ids:
            plan.append(("inspect", saved_ids[0],
                         "Open the concept, skim it, and judge whether it could change the work."))
            plan.append(("surface", saved_ids[0],
                         "Bookmark it for later because it seems worth keeping on the research list."))
        if len(shared_skip_ids) > 1:
            plan.append(("inspect", shared_skip_ids[1],
                         "Check another plausible lead before spending the next bookmark slot."))
            plan.append(("skip", shared_skip_ids[1],
                         "Ignore it — this is the agent deciding a tab is noise rather than signal."))
        for cid in saved_ids[1:]:
            plan.append(("inspect", cid,
                         "Inspect details before deciding whether it deserves one of the limited shortlist slots."))
            plan.append(("surface", cid,
                         "Bookmark it — equivalent to adding it to the reading bucket list."))
        plan.append(("stop", None,
                     "Stop when the reading list is strong enough and grade the whole session."))
        return plan

    def _simulate_session(policy_name, saved_ids, accent, tint, strategy_note):
        session_env = BlindspotEnvironment()
        session_obs = session_env.reset(user_id=uid, seed=0)
        timeline = []
        for action_type, cid, human_note in _build_plan(saved_ids):
            if action_type == "skip":
                record = candidate_by_id.get(cid, {})
                timeline.append({
                    "kind": "skip",
                    "title": record.get("title", f"concept {cid}"),
                    "human_note": human_note,
                    "env_note": "No `surface` action taken. The topic was opened, judged, and then left out of the reading list.",
                    "reward": 0.0,
                    "inspect_budget_remaining": session_obs.inspect_budget_remaining,
                    "surface_budget_remaining": session_obs.surface_budget_remaining,
                    "surfaced_count": len(session_obs.surfaced),
                    "breakdown": None,
                    "record": record,
                    "detail": None,
                })
                continue

            action = BlindspotAction(type=action_type, concept_id=cid if action_type != "stop" else None)
            session_obs = session_env.step(action)
            log_entry = session_env.state.reasoning_log[-1] if session_env.state.reasoning_log else {}
            record = candidate_by_id.get(cid, {}) if cid is not None else None
            detail = session_obs.inspected.get(str(cid)) if action_type == "inspect" and cid is not None else None
            breakdown = None
            if session_obs.reward_breakdown is not None:
                breakdown = {
                    "adoption": session_obs.reward_breakdown.adoption,
                    "novelty": session_obs.reward_breakdown.novelty,
                    "onboarding": session_obs.reward_breakdown.onboarding,
                    "efficiency": session_obs.reward_breakdown.efficiency,
                    "false_positive": session_obs.reward_breakdown.false_positive,
                    "total": session_obs.reward_breakdown.total,
                }
            title = "Stop and grade the shortlist" if action_type == "stop" else record.get("title", f"concept {cid}")
            timeline.append({
                "kind": action_type,
                "title": title,
                "human_note": human_note,
                "env_note": log_entry.get('note', ''),
                "reward": float(log_entry.get('reward', session_obs.reward)),
                "inspect_budget_remaining": session_obs.inspect_budget_remaining,
                "surface_budget_remaining": session_obs.surface_budget_remaining,
                "surfaced_count": len(session_obs.surfaced),
                "breakdown": breakdown,
                "record": record,
                "detail": detail,
            })

        kept_records = [candidate_by_id[cid] for cid in saved_ids if cid in candidate_by_id]
        neglected_records = [
            record for record in candidate_records
            if record["adopted_by_user"] and record["concept_id"] not in saved_ids
        ]
        skipped_records = [candidate_by_id[cid] for cid in shared_skip_ids if cid in candidate_by_id]
        return {
            "name": policy_name,
            "accent": accent,
            "tint": tint,
            "note": strategy_note,
            "saved_ids": saved_ids,
            "timeline": timeline,
            "kept_records": kept_records,
            "neglected_records": neglected_records,
            "skipped_records": skipped_records,
            "score": report['policies'][policy_name]['reward']['total'],
            "adopted_saved": sum(1 for record in kept_records if record["adopted_by_user"]),
            "true_blindspots": sum(1 for record in kept_records
                                     if record["adopted_by_user"] and not record["is_trending"]),
            "trending_noise": sum(1 for record in kept_records
                                    if record["is_trending"] and not record["adopted_by_user"]),
        }

    replay_step_seconds = 1.05
    replay_start_delay = 0.35

    def _render_session_column(session):
        out = []
        out.append(f"<div style='background:white;border:1px solid #eee;border-top:4px solid {session['accent']};"
                   f"border-radius:12px;padding:14px;'>")
        out.append(f"<div style='display:flex;justify-content:space-between;align-items:flex-start;gap:12px;'>"
                   f"<div><div style='font-size:16px;font-weight:800;color:{session['accent']};'>{session['name']}</div>"
                   f"<div style='font-size:12px;color:#666;margin-top:4px;'>{html.escape(session['note'])}</div></div>"
                   f"<div style='font-size:22px;font-weight:800;color:{session['accent']};'>{session['score']:+.2f}</div></div>")
        out.append(f"<div style='font-size:11px;color:#777;margin-top:8px;'>"
                   f"Autoplay replay: {len(session['timeline'])} actions, revealing one action every {replay_step_seconds:.2f}s.</div>")
        out.append("<div style='display:grid;grid-template-columns:repeat(4, 1fr);gap:8px;margin:12px 0 14px 0;'>")
        metrics = [
            ("Bookmarked", f"{len(session['kept_records'])}"),
            ("Later adopted", f"{session['adopted_saved']}"),
            ("True blindspots", f"{session['true_blindspots']}"),
            ("Important topics missed", f"{len(session['neglected_records'])}"),
        ]
        for title, value in metrics:
            out.append(f"<div style='background:{session['tint']};border-radius:8px;padding:8px 10px;'>"
                       f"<div style='font-size:11px;color:#666;'>{title}</div>"
                       f"<div style='font-size:20px;font-weight:800;color:{session['accent']};'>{value}</div>"
                       f"</div>")
        out.append("</div>")
        out.append("<div style='font-size:13px;font-weight:700;margin-bottom:8px;'>Decision trail</div>")
        out.append("<div style='position:relative;padding-left:14px;border-left:3px solid #ddd;'>")
        for event_idx, event in enumerate(session['timeline']):
            record = event.get('record') or {}
            detail = event.get('detail')
            adopted = bool(record.get('adopted_by_user'))
            trend = record.get('is_trending')
            reveal_delay = replay_start_delay + event_idx * replay_step_seconds
            out.append(f"<div class='replay-card' style='position:relative;margin:0 0 12px 0;padding:12px 12px 12px 16px;"
                       f"background:#fafafa;border-radius:10px;border:1px solid #eee;opacity:0;"
                       f"animation: replayReveal 0.5s ease forwards, replayPulse 0.9s ease {reveal_delay + 0.42:.2f}s 1;"
                       f"animation-delay: {reveal_delay:.2f}s, {reveal_delay + 0.42:.2f}s;'>")
            out.append(f"<div style='position:absolute;left:-10px;top:16px;width:14px;height:14px;border-radius:50%;"
                       f"background:{session['accent']};border:3px solid white;box-shadow:0 0 0 1px #ddd;'></div>")
            out.append(_chip(f"Step {event_idx + 1}", "#f5f5f5", "#555"))
            action_label = {
                'inspect': 'Open tab',
                'skip': 'Close / ignore',
                'surface': 'Bookmark',
                'stop': 'Grade session',
            }[event['kind']]
            out.append(_chip(action_label, session['tint'], session['accent']))
            out.append(f"<div style='font-size:14px;font-weight:700;margin-top:7px;'>{html.escape(event['title'])}</div>")
            out.append(f"<div style='font-size:12px;color:#555;margin-top:5px;'><b>Decision:</b> {html.escape(event['human_note'])}</div>")
            if event.get('env_note'):
                out.append(f"<div style='font-size:11px;color:#777;margin-top:4px;'><b>Env:</b> {html.escape(event['env_note'])}</div>")
            if record:
                chips = []
                chips.append(_chip(record.get('topic', 'Other'), '#f5f5f5', '#444'))
                chips.append(_chip('Trending' if trend else 'Under the radar',
                                   '#fff3e0' if trend else '#e8f5e9',
                                   '#ef6c00' if trend else '#2e7d32'))
                chips.append(_chip('Later adopted' if adopted else 'Not adopted later',
                                   '#e8f5e9' if adopted else '#ffebee',
                                   '#2e7d32' if adopted else '#c62828'))
                out.append("<div style='margin-top:5px;'>" + "".join(chips) + "</div>")
            if detail is not None:
                abstract = getattr(detail, 'abstract_summary', '')
                growth_signal = getattr(detail, 'growth_signal', 0.0)
                out.append(f"<div style='margin-top:7px;padding:8px 10px;background:white;border-radius:8px;font-size:11px;color:#666;'>"
                           f"<b>What it saw:</b> {html.escape(abstract[:140])}"
                           f"<div style='margin-top:4px;'>Growth signal: {growth_signal:.2f}</div></div>")
            reward_color = '#22aa66' if event['reward'] > 0 else ('#c62828' if event['reward'] < 0 else '#666')
            out.append(f"<div style='font-size:11px;color:{reward_color};font-weight:700;margin-top:7px;'>"
                       f"step reward {event['reward']:+.2f}</div>")
            out.append(f"<div style='font-size:11px;color:#888;margin-top:3px;'>"
                       f"Inspect left: {event['inspect_budget_remaining']} · "
                       f"Shortlist left: {event['surface_budget_remaining']} · "
                       f"Saved so far: {event['surfaced_count']}</div>")
            if event.get('breakdown'):
                breakdown = event['breakdown']
                out.append(f"<div style='margin-top:8px;padding:8px 10px;background:{session['tint']};border-radius:8px;'>"
                           f"<div style='font-size:11px;color:#555;line-height:1.6;'>"
                           f"Adoption <b>{breakdown['adoption']:+.2f}</b> · "
                           f"Novelty <b>{breakdown['novelty']:+.2f}</b> · "
                           f"Understanding <b>{breakdown['onboarding']:+.2f}</b> · "
                           f"False positives <b>{breakdown['false_positive']:+.2f}</b> · "
                           f"Total <b style='color:{session['accent']};'>{breakdown['total']:+.2f}</b>"
                           f"</div></div>")
            out.append("</div>")
        out.append("</div>")
        out.append("</div>")
        return ''.join(out)

    base_session = _simulate_session(
        "Blindspot (pre-training)",
        pre_ids,
        "#aa4488",
        "#f8eef5",
        "Same starting board, but no Blindspot reward signal. Tends to keep relevance-heavy or trending tabs.",
    )
    rl_session = _simulate_session(
        "Blindspot RL",
        rl_ids,
        "#22aa66",
        "#eef8f1",
        "Same starting board, but trained to maximize adoption + novelty + understanding.",
    )

    topic_rows = []
    for topic, total in topic_counts.most_common(6):
        base_saved = sum(1 for record in base_session['kept_records'] if record['topic'] == topic)
        rl_saved = sum(1 for record in rl_session['kept_records'] if record['topic'] == topic)
        base_neglected = sum(1 for record in base_session['neglected_records'] if record['topic'] == topic)
        rl_neglected = sum(1 for record in rl_session['neglected_records'] if record['topic'] == topic)
        topic_rows.append((topic, total, base_saved, rl_saved, base_neglected, rl_neglected))

    out = []
    out.append("<style>"
               "@keyframes replayReveal {"
               "  0% { opacity: 0; transform: translateY(14px) scale(0.985); filter: blur(2px); }"
               "  100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }"
               "}"
               "@keyframes replayPulse {"
               "  0%, 100% { box-shadow: 0 0 0 rgba(0,0,0,0); }"
               "  50% { box-shadow: 0 0 0 3px rgba(34, 170, 102, 0.14); }"
               "}"
               "@media (prefers-reduced-motion: reduce) {"
               "  .replay-card { opacity: 1 !important; animation: none !important; }"
               "}"
               "</style>")
    out.append("<div style='margin:28px 0;padding:22px;background:#fffdf7;border-radius:14px;"
               "border:1px solid #e8dcc2;'>")
    out.append("<h2 style='margin-top:0;margin-bottom:6px;font-size:18px;'>"
               "🎮 Same session, two agents — base model vs RL policy</h2>")
    out.append("<p style='color:#666;font-size:13px;margin-top:0;margin-bottom:18px;'>"
               "Think of this like a Mario RL rollout, but for research. Both agents start from the same board, see the same tempting distractions, "
               "and have the same budgets. The only difference is what they decide to bookmark. That makes the behavioral change from SFT training visible instead of abstract.</p>")
    out.append(f"<div style='margin-bottom:18px;padding:10px 12px;background:#f8fafc;border:1px solid #e5e7eb;border-radius:10px;font-size:12px;color:#555;'>"
               f"▶️ Autoplay is on. Both columns reveal the same action index together every {replay_step_seconds:.2f}s,"
               " so the audience can watch the two policies diverge step by step.</div>")

    out.append("<div style='display:grid;grid-template-columns:repeat(4, 1fr);gap:12px;margin-bottom:18px;'>")
    stat_cards = [
        ("Candidate tabs", f"{len(candidate_records)}", "Topics in the starting board"),
        ("Trending distractions", f"{total_trending}", "Easy-to-find popular topics"),
        ("Under-the-radar topics", f"{total_under_radar}", "Potential blindspots outside the obvious feed"),
        ("Actually mattered later", f"{total_adopted}", "Topics the matched researcher later adopted"),
    ]
    for title, value, subtitle in stat_cards:
        out.append(f"<div style='background:white;border:1px solid #eee;border-radius:10px;padding:14px;'>"
                   f"<div style='font-size:12px;color:#888;margin-bottom:6px;'>{title}</div>"
                   f"<div style='font-size:28px;font-weight:800;line-height:1;'>{value}</div>"
                   f"<div style='font-size:12px;color:#666;margin-top:8px;'>{subtitle}</div>"
                   f"</div>")
    out.append("</div>")

    out.append("<div style='display:grid;grid-template-columns:1.1fr 0.9fr;gap:14px;margin-bottom:18px;'>")
    out.append("<div style='background:white;border:1px solid #eee;border-radius:10px;padding:14px;'>")
    out.append(f"<div style='font-size:14px;font-weight:700;margin-bottom:8px;'>Starting board for matched researcher <code>{html.escape(uid)}</code></div>")
    out.append(f"<div style='font-size:12px;color:#666;margin-bottom:10px;'>First few tabs in the pool: {', '.join(start_titles)}</div>")
    out.append("<div style='font-size:12px;color:#555;'>")
    for topic, total in topic_counts.most_common(8):
        out.append(_chip(f"{topic}: {total}", "#f5f5f5", "#444"))
    out.append("</div></div>")

    out.append("<div style='background:white;border:1px solid #eee;border-radius:10px;padding:14px;'>")
    out.append("<div style='font-size:14px;font-weight:700;margin-bottom:8px;'>Topic coverage map</div>")
    out.append("<table style='width:100%;border-collapse:collapse;font-size:12px;'>")
    out.append("<tr style='background:#f7f7f7;'><th style='text-align:left;padding:6px;'>Topic</th>"
               "<th style='padding:6px;'>Pool</th><th style='padding:6px;'>Base saved</th>"
               "<th style='padding:6px;'>RL saved</th><th style='padding:6px;'>Base neglected</th>"
               "<th style='padding:6px;'>RL neglected</th></tr>")
    for topic, total, base_saved, rl_saved, base_neglected, rl_neglected in topic_rows:
        out.append(f"<tr style='border-top:1px solid #f0f0f0;'>"
                   f"<td style='padding:6px;text-align:left;'>{html.escape(topic)}</td>"
                   f"<td style='padding:6px;text-align:center;'>{total}</td>"
                   f"<td style='padding:6px;text-align:center;color:#aa4488;font-weight:700;'>{base_saved}</td>"
                   f"<td style='padding:6px;text-align:center;color:#22aa66;font-weight:700;'>{rl_saved}</td>"
                   f"<td style='padding:6px;text-align:center;color:#aa4488;font-weight:700;'>{base_neglected}</td>"
                   f"<td style='padding:6px;text-align:center;color:#22aa66;font-weight:700;'>{rl_neglected}</td></tr>")
    out.append("</table></div>")
    out.append("</div>")

    out.append("<h3 style='margin-bottom:10px;font-size:15px;'>🧩 Base model plays the same session vs RL policy plays the same session</h3>")
    out.append("<p style='font-size:13px;color:#666;margin-top:-4px;margin-bottom:14px;'>"
               "Both columns start from the same board and even waste attention on the same distraction tabs. "
               "The behavior difference comes from what each model decides to bookmark and what it leaves neglected."
               " The cards below autoplay in sync like a replay.</p>")
    out.append("<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px;'>")
    out.append(_render_session_column(base_session))
    out.append(_render_session_column(rl_session))
    out.append("</div>")

    out.append("<h3 style='margin-top:22px;margin-bottom:10px;font-size:15px;'>🗂️ What each agent covered vs neglected</h3>")
    out.append("<div style='display:grid;grid-template-columns:1fr 1fr;gap:14px;'>")
    for session in [base_session, rl_session]:
        out.append(f"<div style='background:white;border:1px solid #eee;border-top:4px solid {session['accent']};border-radius:12px;padding:12px;'>")
        out.append(f"<div style='font-size:14px;font-weight:800;color:{session['accent']};margin-bottom:8px;'>{session['name']}</div>")
        out.append("<div style='font-size:12px;color:#666;margin-bottom:8px;'>Bookmarked in this session</div>")
        if session['kept_records']:
            for record in session['kept_records'][:3]:
                out.append(_mini_topic_card(record, session['accent'], session['tint'], 'saved'))
        else:
            out.append("<div style='font-size:12px;color:#999;padding:6px 0;'>No concepts bookmarked.</div>")
        out.append("<div style='font-size:12px;color:#666;margin:12px 0 8px 0;'>Opened then ignored</div>")
        for record in session['skipped_records'][:2]:
            out.append(_mini_topic_card(record, '#ef6c00', '#fff3e0', 'closed'))
        out.append("<div style='font-size:12px;color:#666;margin:12px 0 8px 0;'>Neglected but later mattered</div>")
        if session['neglected_records']:
            for record in session['neglected_records'][:3]:
                out.append(_mini_topic_card(record, '#c62828', '#ffebee', 'missed blindspot'))
        else:
            out.append("<div style='font-size:12px;color:#999;padding:6px 0;'>None in this sample.</div>")
        out.append("</div>")
    out.append("</div>")

    out.append("</div>")
    return '\n'.join(out)


# ──────────────────────────── RL visual sections ────────────────────────────
def render_rl_visual(report):
    """Three visual panels that show what RL actually learned:
    1. Before vs After diff — which concepts changed and why
    2. Per-concept reward trace — each pick graded against real ground truth
    3. Full adoption matrix — which policy found which adopted concept
    """
    policies = report['policies']
    pre = policies.get('Blindspot (pre-training)', {})
    rl  = policies.get('Blindspot RL', {})
    if not pre or not rl:
        return ""

    out = []
    out.append("<div style='margin:28px 0;padding:22px;background:#f8f9fa;"
               "border-radius:14px;border:1px solid #dde;'>")
    out.append("<h2 style='margin-top:0;margin-bottom:4px;font-size:18px;'>"
               "🔬 After many episodes like that, what did SFT training learn?</h2>")
    out.append("<p style='color:#666;font-size:13px;margin-bottom:24px;'>"
               "The section above showed the same session played by the base model and the RL policy. "
               "This section is the audit afterwards: same researcher, same candidate pool, "
               "but now comparing the final outputs and rewards side by side.</p>")

    # ── Build card lookup ──
    card_lookup = {}
    for res in policies.values():
        for card in res.get('cards', []):
            card_lookup[card['concept_id']] = card

    pre_ids = list(pre.get('surfaced', []))
    rl_ids  = list(rl.get('surfaced', []))
    pre_set = set(pre_ids)
    rl_set  = set(rl_ids)

    # ─────────────────────────────────────────────────────────────────
    # PANEL 1: Side-by-side before/after diff
    # ─────────────────────────────────────────────────────────────────
    out.append("<h3 style='margin-bottom:10px;font-size:15px;'>"
               "1️⃣ &nbsp;What changed after SFT training?</h3>")
    out.append("<p style='color:#666;font-size:13px;margin-top:-4px;margin-bottom:14px;'>"
               "Green card = researcher actually adopted this concept. "
               "Red card = wasted recommendation (not adopted).</p>")
    out.append("<div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:28px;'>")

    def _concept_card_mini(cid, tag_html=""):
        card = card_lookup.get(cid)
        if not card:
            return f"<div style='margin:4px 0;padding:8px;background:#eee;border-radius:6px;font-size:12px;'>{cid}</div>"
        hit = card['adopted_by_user']
        bg   = '#e8f5e9' if hit else '#fce8e8'
        border = '#22aa66' if hit else '#cc3333'
        icon = '✅' if hit else '❌'
        trend = '🔥 Trending' if card['is_trending'] else '💎 Under the radar'
        return (f"<div style='margin:5px 0;padding:10px 12px;background:{bg};"
                f"border-left:4px solid {border};border-radius:0 7px 7px 0;'>"
                f"<div style='font-size:13px;font-weight:600;'>{icon} {html.escape(card['title'][:48])}</div>"
                f"<div style='font-size:11px;color:#777;margin-top:3px;'>{trend}{tag_html}</div>"
                f"</div>")

    # Before column
    out.append("<div>")
    pre_total = pre.get('reward', {}).get('total', 0)
    out.append(f"<div style='font-weight:700;font-size:14px;color:#aa4488;margin-bottom:8px;'>"
               f"Base model (before SFT) &nbsp; "
               f"<span style='background:#aa448820;padding:2px 8px;border-radius:12px;font-size:13px;'>"
               f"Score: {pre_total:+.2f}</span></div>")
    for cid in pre_ids:
        dropped = cid not in rl_set
        tag = ' &nbsp;<span style="background:#cc333320;color:#cc3333;padding:1px 5px;border-radius:4px;font-size:10px;">dropped by RL</span>' if dropped else \
              ' &nbsp;<span style="background:#22aa6620;color:#22aa66;padding:1px 5px;border-radius:4px;font-size:10px;">kept by RL</span>'
        out.append(_concept_card_mini(cid, tag))
    out.append("</div>")

    # After column
    out.append("<div>")
    rl_total = rl.get('reward', {}).get('total', 0)
    out.append(f"<div style='font-weight:700;font-size:14px;color:#22aa66;margin-bottom:8px;'>"
               f"After SFT training &nbsp; "
               f"<span style='background:#22aa6620;padding:2px 8px;border-radius:12px;font-size:13px;'>"
               f"Score: {rl_total:+.2f}</span></div>")
    for cid in rl_ids:
        is_new = cid not in pre_set
        card = card_lookup.get(cid, {})
        hit = card.get('adopted_by_user', False)
        if is_new and hit:
            tag = ' &nbsp;<span style="background:#22aa6620;color:#22aa66;padding:1px 5px;border-radius:4px;font-size:10px;font-weight:700;">★ new discovery</span>'
        elif is_new:
            tag = ' &nbsp;<span style="background:#cc773320;color:#cc7733;padding:1px 5px;border-radius:4px;font-size:10px;">new pick</span>'
        else:
            tag = ' &nbsp;<span style="background:#3377cc20;color:#3377cc;padding:1px 5px;border-radius:4px;font-size:10px;">kept</span>'
        out.append(_concept_card_mini(cid, tag))
    out.append("</div>")
    out.append("</div>")  # end grid

    # ─────────────────────────────────────────────────────────────────
    # PANEL 2: Per-concept reward trace (RL policy)
    # ─────────────────────────────────────────────────────────────────
    out.append("<h3 style='margin-bottom:8px;font-size:15px;'>"
               "2️⃣ &nbsp;RL reward earned — concept by concept</h3>")
    out.append("<p style='color:#666;font-size:13px;margin-top:-4px;margin-bottom:14px;'>"
               "This is the exact reward signal the model was trained on. "
               "Every bar = one concept the RL model surfaced. "
               "Green = the researcher adopted it (the model got paid). "
               "Red = recommendation wasted (penalty).</p>")

    out.append("<div style='background:white;padding:14px 16px;border-radius:10px;"
               "border:1px solid #eee;margin-bottom:24px;'>")
    for card in rl.get('cards', []):
        adopted = card['adopted_by_user']
        novel   = not card['is_trending']
        comp    = card.get('comprehension_lift', 0.0)
        r_adopt = 1.0 if adopted else -0.1
        r_novel = 0.5 if (adopted and novel) else 0.0
        r_comp  = min(comp, 1.0) if adopted else 0.0
        r_total = r_adopt + r_novel + r_comp
        bar_max = 200
        bar_w = max(4, int(min(abs(r_total) / 2.5, 1.0) * bar_max))
        bar_color = '#22aa66' if r_total > 0 else '#cc3333'
        adopted_txt = "✅ Adopted by this researcher" if adopted else "❌ Not adopted — penalty"
        components = []
        if r_adopt > 0:
            components.append(f"+{r_adopt:.1f} adoption")
        else:
            components.append(f"{r_adopt:.1f} penalty")
        if r_novel > 0:
            components.append(f"+{r_novel:.1f} novelty bonus")
        if r_comp > 0:
            components.append(f"+{r_comp:.2f} comprehension")
        component_str = " · ".join(components)
        out.append(f"<div style='display:flex;align-items:center;gap:12px;margin:8px 0;flex-wrap:wrap;'>")
        out.append(f"<div style='width:190px;font-size:13px;font-weight:600;flex-shrink:0;'>"
                   f"{html.escape(card['title'][:36])}</div>")
        out.append(f"<div style='flex-shrink:0;'>"
                   f"<div style='display:inline-block;height:20px;width:{bar_w}px;"
                   f"background:{bar_color};border-radius:3px;vertical-align:middle;'></div>"
                   f"<span style='font-size:13px;color:{bar_color};font-weight:700;"
                   f"margin-left:6px;'>{r_total:+.1f}</span></div>")
        out.append(f"<div style='font-size:12px;color:#666;'>{adopted_txt} "
                   f"<span style='color:#aaa;'>({component_str})</span></div>")
        out.append("</div>")

    # Missed adopted concepts — things the researcher adopted but RL didn't surface
    all_adopted_in_pool = {c['concept_id']: c['title']
                           for res in policies.values()
                           for c in res.get('cards', [])
                           if c['adopted_by_user']}
    rl_missed_adopted = {cid: title for cid, title in all_adopted_in_pool.items()
                         if cid not in rl_set}
    if rl_missed_adopted:
        out.append("<div style='margin-top:12px;padding:10px 12px;background:#fff8e1;"
                   "border-left:4px solid #ffb300;border-radius:0 6px 6px 0;'>")
        out.append("<div style='font-size:13px;font-weight:600;color:#e65100;margin-bottom:6px;'>"
                   "🔴 These were in the pool, adopted by the researcher, but RL missed them:</div>")
        for cid, title in rl_missed_adopted.items():
            out.append(f"<div style='font-size:12px;color:#666;'>• {html.escape(title[:60])}</div>")
        out.append("<div style='font-size:11px;color:#999;margin-top:6px;'>"
                   "These are the true unknown-unknowns the model failed to surface — "
                   "room for the next training iteration to improve.</div>")
        out.append("</div>")
    out.append("</div>")  # end panel 2 card

    # ─────────────────────────────────────────────────────────────────
    # PANEL 3: Full policy hit matrix
    # ─────────────────────────────────────────────────────────────────
    out.append("<h3 style='margin-bottom:8px;font-size:15px;'>"
               "3️⃣ &nbsp;Full adoption matrix — who found what?</h3>")
    out.append("<p style='color:#666;font-size:13px;margin-top:-4px;margin-bottom:14px;'>"
               "Every concept this researcher <em>actually adopted</em> (rows). "
               "Whether each strategy surfaced it (columns). "
               "✅ = found the blindspot · blank = missed it.</p>")

    if all_adopted_in_pool:
        short_names = {
            'Random': 'Random',
            'Trending': 'Trending',
            'Dense Retrieval': 'Dense',
            'Blindspot (pre-training)': 'Before SFT',
            'Blindspot RL': 'After SFT ⭐',
        }
        policy_names = list(policies.keys())
        policy_surfaced_sets = {name: set(res.get('surfaced', [])) for name, res in policies.items()}

        out.append("<div style='overflow-x:auto;background:white;padding:14px;border-radius:10px;"
                   "border:1px solid #eee;'>")
        out.append("<table style='border-collapse:collapse;font-size:12px;min-width:600px;'>")
        out.append("<tr style='background:#f0f0f0;'>")
        out.append("<th style='padding:7px 12px;text-align:left;min-width:200px;'>Adopted concept</th>")
        for name in policy_names:
            color = PALETTE.get(name, '#666')
            short = short_names.get(name, name)
            out.append(f"<th style='padding:7px 10px;text-align:center;color:{color};'>{short}</th>")
        out.append("</tr>")

        for cid, title in sorted(all_adopted_in_pool.items(), key=lambda x: x[1]):
            out.append("<tr style='border-bottom:1px solid #f0f0f0;'>")
            out.append(f"<td style='padding:6px 12px;font-weight:500;'>{html.escape(title[:50])}</td>")
            for name in policy_names:
                found = cid in policy_surfaced_sets.get(name, set())
                cell_bg = '#e8f5e9' if found else 'white'
                cell = '✅' if found else '<span style="color:#ddd;">—</span>'
                out.append(f"<td style='padding:6px 10px;text-align:center;background:{cell_bg};'>{cell}</td>")
            out.append("</tr>")

        # Summary row: hit rate per policy
        out.append("<tr style='background:#f8f8f8;border-top:2px solid #ddd;font-weight:700;'>")
        out.append("<td style='padding:7px 12px;'>Hit rate</td>")
        total_adopted = len(all_adopted_in_pool)
        for name in policy_names:
            found_count = sum(1 for cid in all_adopted_in_pool if cid in policy_surfaced_sets.get(name, set()))
            pct = found_count / total_adopted * 100 if total_adopted else 0
            color = PALETTE.get(name, '#666')
            out.append(f"<td style='padding:7px 10px;text-align:center;color:{color};'>"
                       f"{found_count}/{total_adopted} ({pct:.0f}%)</td>")
        out.append("</tr>")
        out.append("</table></div>")

    out.append("</div>")  # end outer card
    return '\n'.join(out)


# ──────────────────────────── HTML render ────────────────────────────
def render_html(report, focus="Blindspot RL"):
    p = report['profile']
    out = ["<div style='font-family:-apple-system,sans-serif;max-width:1100px;'>"]

    mode = p.get('mode', 'paragraph-match')
    mode_text = {
        'real-user': '🟢 Real researcher — rewards measured against actual adoption history',
        'persona': '🟣 Researcher archetype — matched to closest real profile in database',
        'paragraph-match': '🔵 Your background — matched to closest researcher in database',
    }.get(mode, mode)
    if p.get('weak_match_warning'):
        mode_text += ' · ⚠️ low similarity match — try Tab 1 for stronger ground truth'

    out.append("<div style='background:#f0f4f8;padding:14px;border-radius:8px;margin-bottom:20px;'>")
    out.append(f"<div style='font-size:14px;font-weight:600;margin-bottom:6px;'>{mode_text}</div>")
    out.append(f"Scanning <b>{p['candidate_pool_size']} candidate concepts</b> for researcher "
               f"<code>{p['matched_user_id']}</code> (similarity: <b>{p['match_similarity']:.2f}</b>)<br>")
    if p.get('shared_keywords'):
        kws = ', '.join(f"<code>{html.escape(k)}</code>" for k in p['shared_keywords'][:8])
        out.append(f"<small>Profile matched on: {kws}</small><br>")
    out.append(f"<small style='color:#555;margin-top:4px;display:block;'>"
               f"Researcher summary: {html.escape(p['matched_summary'][:240])}…</small></div>")

    policies = report['policies']

    # Human-style episode replay using the real environment actions.
    out.append(render_human_research_loop(report))

    # Headline before/after lift
    before_r = policies.get('Blindspot (pre-training)', {}).get('reward', {}).get('total', 0)
    after_r  = policies.get('Blindspot RL', {}).get('reward', {}).get('total', 0)
    lift = after_r - before_r
    lift_color = '#22aa66' if lift > 0 else '#cc3333'
    lift_emoji = '📈' if lift > 0 else '📉'
    lift_story = ("RL training made recommendations significantly better" if lift > 1
                  else "RL training improved recommendations" if lift > 0
                  else "RL training is still learning for this profile")
    out.append(f"<div style='background:#e8f5e9;padding:16px;border-radius:10px;"
               f"margin-bottom:20px;text-align:center;'>"
               f"<div style='font-size:13px;color:#555;margin-bottom:6px;'>Did SFT training help for this researcher?</div>"
               f"<div style='font-size:20px;'>"
               f"Base model (before RL): <span style='color:#aa4488;font-weight:700;'>{before_r:+.2f}</span>"
               f" &nbsp;→&nbsp; "
               f"After SFT training: <span style='color:#22aa66;font-weight:700;'>{after_r:+.2f}</span>"
               f" &nbsp; {lift_emoji} <span style='color:{lift_color};font-weight:700;'>{'+' if lift>=0 else ''}{lift:.2f}</span>"
               f"</div>"
               f"<div style='font-size:12px;color:#666;margin-top:6px;'>{lift_story}</div>"
               f"</div>")

    # RL visual panels: diff, reward trace, adoption matrix
    out.append(render_rl_visual(report))

    # Reward bar chart with human labels
    max_r = max(abs(v['reward']['total']) for v in policies.values()) or 1
    out.append("<h3 style='margin-bottom:6px;'>📊 How each strategy scored</h3>")
    out.append("<p style='color:#666;font-size:13px;margin-top:0;'>Higher = found more concepts this researcher actually adopted + understood. "
               "Random ≈ 0 confirms the reward signal is calibrated.</p>")
    out.append("<table style='width:100%;border-collapse:collapse;'>")
    for name, res in policies.items():
        r = res['reward']['total']
        bar_w = int(abs(r) / max_r * 280)
        color = PALETTE.get(name, '#666')
        latency = res.get('latency_ms', 0)
        is_winner = (name == focus)
        row_bg = "background:#f0fdf4;" if is_winner else ""
        star = " ⭐" if is_winner else ""
        out.append(f"<tr style='{row_bg}'>"
                   f"<td style='width:210px;padding:7px 6px;font-size:13px;'><b>{name}{star}</b></td>"
                   f"<td style='width:70px;padding:7px 6px;font-weight:700;color:{color};'>{r:+.2f}</td>"
                   f"<td style='width:300px;padding:7px 0;'>"
                   f"<div style='display:inline-block;width:{bar_w}px;height:16px;background:{color};border-radius:3px;'></div>"
                   f"</td>"
                   f"<td style='color:#aaa;font-size:11px;padding:7px 6px;'>⚡{latency:.0f}ms</td></tr>")
    out.append("</table>")

    out.append("<h3 style='margin-top:28px;margin-bottom:6px;'>🔍 What was surfaced — and did it help?</h3>")
    out.append("<p style='color:#666;font-size:13px;margin-top:0;'>"
               "Each concept card shows: did the researcher adopt it? did it improve understanding? "
               "Green border = adopted ✅ · Red border = not adopted ❌</p>")
    for name, res in policies.items():
        is_open = (name == focus)
        cache_note = ''
        if res['meta'].get('used_cache'):
            cache_note = ' <span style="font-size:11px;color:#888;">💾 pre-cached response</span>'
        elif res['meta'].get('used_trained_model'):
            cache_note = ' <span style="font-size:11px;color:#888;">🤖 live model</span>'
        if res['meta'].get('used_nearest_neighbor'):
            cache_note += f" <span style='font-size:11px;color:#888;'>🔁 nearest match: {res['meta']['used_nearest_neighbor']}</span>"

        r = res['reward']
        palette_color = PALETTE.get(name, '#666')
        out.append(f"<details {'open' if is_open else ''} "
                   f"style='margin:10px 0;border-left:4px solid {palette_color};"
                   f"padding-left:14px;border-radius:0 6px 6px 0;background:#fafafa;padding:10px 10px 10px 14px;'>")
        out.append(f"<summary style='cursor:pointer;font-size:14px;'>"
                   f"<b>{name}</b>{cache_note} — "
                   f"Score: <b style='color:{palette_color};'>{r['total']:+.2f}</b> "
                   f"<span style='color:#888;font-size:12px;'>"
                   f"(adopted {r['adoption']:+.2f} · novel {r['novelty']:+.2f} · "
                   f"understood {r['onboarding']:+.2f} · wasted {r['false_positive']:+.2f})"
                   f"</span></summary>")
        out.append(f"<p style='color:#555;font-size:13px;font-style:italic;'>{html.escape(res['description'])}</p>")
        for card in res['cards']:
            color = '#22aa66' if card['adopted_by_user'] else '#cc3333'
            adopted_label = "✅ This researcher actually adopted this concept" if card['adopted_by_user'] else "❌ Not adopted by this researcher"
            out.append(f"<div style='margin:10px 0;padding:12px 14px;background:white;"
                       f"border-left:4px solid {color};border-radius:0 6px 6px 0;box-shadow:0 1px 3px rgba(0,0,0,0.06);'>")
            tag = '🔥 Trending' if card['is_trending'] else '💎 Under the radar'
            out.append(f"<div style='font-weight:700;font-size:14px;margin-bottom:4px;'>"
                       f"{html.escape(card['title'])} "
                       f"<span style='font-size:11px;font-weight:400;color:#888;'>[{card['concept_id']}]</span></div>")
            out.append(f"<div style='font-size:13px;color:#555;margin-bottom:6px;'>{html.escape(card['one_liner'][:160])}</div>")
            tags = [tag, adopted_label]
            if card['comprehension_lift'] > 0:
                tags.append(f"📈 Understanding improved by {card['comprehension_lift']:+.2f} after reading")
            out.append(f"<div style='font-size:12px;'>" + " &nbsp;·&nbsp; ".join(tags) + "</div>")
            if card.get('verdict'):
                out.append(f"<div style='font-size:12px;color:#444;margin-top:4px;'><b>Why surfaced:</b> {html.escape(card['verdict'])}</div>")
            if card['reading_path']:
                out.append("<details style='margin-top:8px;'><summary style='font-size:12px;cursor:pointer;color:#3377cc;'>📚 5-paper reading path to learn this</summary><ol style='font-size:12px;margin-top:6px;'>")
                for paper in card['reading_path'][:5]:
                    out.append(f"<li>{html.escape(paper.get('title','?'))} ({paper.get('year','?')})</li>")
                out.append("</ol></details>")
            out.append("</div>")
        if res['meta'].get('reasoning'):
            out.append(f"<details style='margin-top:8px;'><summary style='font-size:12px;cursor:pointer;'>💭 See model reasoning</summary>"
                       f"<pre style='white-space:pre-wrap;font-size:12px;'>"
                       f"{html.escape(res['meta']['reasoning'])}</pre></details>")
        out.append("</details>")

    if report.get('chatgpt_baseline'):
        out.append("<h3 style='margin-top:28px;'>💬 What a generic LLM would say instead</h3>")
        out.append("<p style='color:#666;font-size:13px;'>For comparison — a generic ChatGPT prompt with no knowledge of this person's catalog or adoption history:</p>")
        out.append(f"<div style='background:#fff3e0;padding:14px;border-radius:8px;border-left:4px solid #ff9800;'>"
                   f"{html.escape(report['chatgpt_baseline'])}</div>")
        out.append("<p style='font-size:12px;color:#888;margin-top:6px;'>"
                   "↑ Generic advice. Blindspot instead picks from a specific 1,168-concept catalog "
                   "and validates every recommendation against real adoption + comprehension data.</p>")

    out.append("</div>")
    return '\n'.join(out)


# ──────────────────────────── concept browser ────────────────────────────
def render_catalog(filter_text: str = "", show_only: str = "all"):
    cat = demo_engine.catalog
    cids = list(cat.keys())
    if filter_text:
        ft = filter_text.lower()
        cids = [c for c in cids if ft in (cat[c].get('title', '') + ' '
                                          + cat[c].get('one_liner', '')).lower()]
    if show_only == "trending":
        cids = [c for c in cids if cat[c].get('is_trending')]
    elif show_only == "novel":
        cids = [c for c in cids if not cat[c].get('is_trending')]

    cids = cids[:200]
    out = [f"<div style='max-width:1000px;'><h3>📚 Concept catalog · {len(cids)} of "
           f"{len(cat)} shown</h3>"]
    out.append("<table style='width:100%;border-collapse:collapse;font-size:13px;'>")
    out.append("<tr style='background:#eee;'><th>id</th><th>title</th><th>one-liner</th>"
               "<th>tag</th><th>growth</th></tr>")
    for c in cids:
        r = cat[c]
        tag = '🔥' if r.get('is_trending') else '💎'
        out.append(f"<tr><td>{c}</td><td><b>{html.escape(r.get('title','?')[:60])}</b></td>"
                   f"<td>{html.escape(r.get('one_liner','')[:80])}</td>"
                   f"<td>{tag}</td><td>{r.get('growth_signal',0):.2f}</td></tr>")
    out.append("</table></div>")
    return '\n'.join(out)


# ──────────────────────────── Gradio app ────────────────────────────

INTRO = """
# 🧠 Blindspot — Find What You Don't Know You're Missing

---

**Every day you do this:**

> You open Twitter. You skim some papers. You read a blog post. You bookmark a few things.
> You feel caught up. But are you?

The problem is: **you only find what you already know to search for.**

The concepts that would actually change your research direction?
The ones just outside your radar? The ones your peers adopted 6 months before you?

**Those are your blindspots. And you'll never find them by searching.**

---

### 💡 What Blindspot does

Blindspot is an AI trained with **SFT (Supervised Fine-Tuning)** to act like a research assistant that:

| Step | What happens |
|------|-------------|
| **Reads your profile** | Understands your current work and past papers |
| **Scans 1,168 ML concepts** | All in 3ms — what you'd take weeks to browse manually |
| **Decides what to surface** | Not just trending — specifically what *you* will adopt and understand |
| **Shows you the proof** | Every recommendation checked against real adoption ground truth |

---

### 🔬 The RL training explained simply

The AI learned by doing what you do — inspect a topic → decide to keep or skip → stop when done.

**Reward signal:**
- ✅ +1 if you would have adopted it
- 📈 +0.5 if it improves your understanding
- 🔥 +0.5 if it's novel to you specifically
- ❌ −0.1 for every useless recommendation

**After 3,200 training episodes across 13 real ML researchers:**
the AI learned to beat every baseline — including the "just show trending" approach.

---

**👇 Start with Tab 1 → Pick a researcher → See what the AI finds for them**
"""

STEP1_GUIDE = """
### Step 1 — Pick a real ML researcher

These are 17 actual researchers in our database (we tracked what concepts they adopted over time).

**What you'll see:**
- 5 different strategies tried on the same person
- Each strategy scored: did they actually adopt what was recommended?
- **The key comparison:** AI before training vs AI after SFT training

> 💡 **Start with the default user, click Run, and scroll down**
"""

STEP2_GUIDE = """
### Step 2 — Try a researcher archetype

Don't have a specific researcher? Pick a role you relate to.

Each persona is a pre-written description (e.g. "I'm a diffusion models PhD student...").
The AI finds the closest real researcher in our database and shows what it would surface for them.

> 💡 **Try "LLM Agents Researcher" — then toggle Before/After to see the RL improvement**
"""

STEP3_GUIDE = """
### Step 3 — Try with your own background

Paste a short description of your work (LinkedIn bio, research summary, anything).

The AI will find the closest researcher in our database and show you what concepts
it would surface for someone like you.

> 💡 **This shows how the system would work for a new user it has never seen before**
"""


PERSONA_LABELS = {
    "🤖 LLM Agents Researcher (frontier lab)": "llm_agents_researcher",
    "🔬 Diffusion Models PhD Student":         "diffusion_phd_student",
    "⚙️ ML Infra Engineer (50-person startup)": "ml_infra_engineer",
}

def run_real_user(uid, focus_label):
    focus = "Blindspot RL" if focus_label == "After SFT training ✅" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(user_id=uid), focus=focus)

def run_persona(persona_label, focus_label):
    key = PERSONA_LABELS.get(persona_label)
    if not key:
        return "<i>Unknown persona</i>"
    focus = "Blindspot RL" if focus_label == "After SFT training ✅" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(paragraph=PERSONAS[key], persona_key=key),
                       focus=focus)

def run_paragraph(text, focus_label):
    if not text or not text.strip():
        return "<i>Paste a paragraph about your work and click Run.</i>"
    focus = "Blindspot RL" if focus_label == "After SFT training ✅" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(paragraph=text), focus=focus)


LIGHT_CSS = """
:root { color-scheme: light !important; }
body, .gradio-container, .main, .wrap, .app {
    background: #f7f8fa !important;
    color: #111 !important;
}
.dark, [data-testid], .gr-box, .gr-panel, .prose,
.svelte-1ed2p3z, .svelte-phx3hq {
    background: #ffffff !important;
    color: #111 !important;
}
.gr-output, .output-html, .gr-html, [class*="output"] {
    background: #ffffff !important;
    color: #111 !important;
}
textarea, input, .gr-input { background: #fff !important; color: #111 !important; }
.tab-nav button { color: #333 !important; }
.tab-nav button.selected { color: #4f46e5 !important; border-bottom-color: #4f46e5 !important; }
"""

with gr.Blocks(
    title="Blindspot — Unknown-Unknowns Discovery",
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=LIGHT_CSS,
) as ui:
    gr.Markdown(INTRO)

    with gr.Tabs():

        # ── TAB 1: Real researcher ──────────────────────────────────────────
        with gr.TabItem("1️⃣ Real Researcher (start here)"):
            gr.Markdown(STEP1_GUIDE)
            with gr.Row():
                real_user = gr.Dropdown(
                    choices=USER_OPTIONS,
                    label="👤 Pick a researcher",
                    value=USER_OPTIONS[0][1],
                    scale=4,
                )
                focus1 = gr.Radio(
                    choices=["After SFT training ✅", "Before training (base model)"],
                    value="After SFT training ✅",
                    label="🔀 Show results from",
                    scale=2,
                )
            btn1 = gr.Button("🔍 Find their blindspots", variant="primary", size="lg")
            out1 = gr.HTML()
            btn1.click(run_real_user, inputs=[real_user, focus1], outputs=out1)

        # ── TAB 2: Persona ──────────────────────────────────────────────────
        with gr.TabItem("2️⃣ Try a Persona"):
            gr.Markdown(STEP2_GUIDE)
            with gr.Row():
                persona = gr.Radio(
                    choices=list(PERSONA_LABELS.keys()),
                    value=list(PERSONA_LABELS.keys())[0],
                    label="🎭 I am a...",
                    scale=4,
                )
                focus2 = gr.Radio(
                    choices=["After SFT training ✅", "Before training (base model)"],
                    value="After SFT training ✅",
                    label="🔀 Show results from",
                    scale=2,
                )
            btn2 = gr.Button("🔍 Find my blindspots", variant="primary", size="lg")
            out2 = gr.HTML()
            btn2.click(run_persona, inputs=[persona, focus2], outputs=out2)

        # ── TAB 3: Your own paragraph ───────────────────────────────────────
        with gr.TabItem("3️⃣ Your Own Background"):
            gr.Markdown(STEP3_GUIDE)
            with gr.Row():
                text = gr.Textbox(
                    lines=5,
                    placeholder="e.g. I work on LLM agents, focusing on tool use and planning. "
                                "Recent papers include work on multi-step reasoning and code generation...",
                    label="📝 Describe your work",
                    scale=4,
                )
                focus3 = gr.Radio(
                    choices=["After SFT training ✅", "Before training (base model)"],
                    value="After SFT training ✅",
                    label="🔀 Show results from",
                    scale=2,
                )
            btn3 = gr.Button("🔍 Find my blindspots", variant="primary", size="lg")
            out3 = gr.HTML()
            btn3.click(run_paragraph, inputs=[text, focus3], outputs=out3)

        # ── TAB 4: Concept catalog ──────────────────────────────────────────
        with gr.TabItem("4️⃣ Browse Concept Catalog"):
            gr.Markdown("""
### The 1,168 concepts Blindspot picks from

This is the full catalog of ML/AI concepts the system scans for every researcher.
You'd need weeks to browse these manually. The AI scans all of them in 3ms.

- 🔥 **Trending** = high adoption growth across researchers recently
- 💎 **Novel** = lower visibility but potentially high value for the right person
""")
            with gr.Row():
                ft = gr.Textbox(label="🔎 Search by keyword", placeholder="e.g. diffusion, agents, scaling")
                so = gr.Radio(choices=["all", "trending", "novel"], value="all",
                              label="Filter")
            btn4 = gr.Button("Search", variant="primary")
            out4 = gr.HTML(value=render_catalog())
            btn4.click(render_catalog, inputs=[ft, so], outputs=out4)

        # ── TAB 5: How it works ─────────────────────────────────────────────
        with gr.TabItem("5️⃣ How It Works"):
            gr.Markdown("""
### The full picture

---

#### 🔄 What happens when you click "Find my blindspots"

```
Your profile / paragraph
        ↓
Matched to closest researcher in our database (TF-IDF cosine similarity)
        ↓
5 strategies run on their 40-concept candidate pool:

    Random        → picks 3 at random
    Trending      → picks 3 most popular
    Dense         → picks 3 most similar to your past work
    Pre-training  → Qwen2.5-1.5B base model picks (no SFT)
    SFT trained   → Qwen2.5-1.5B + LoRA (SFT on 40 expert traces) ← this is Blindspot
                ↓
Each strategy scored:
  ✅ Did the researcher actually adopt this concept? (+reward)
  📈 Did it improve their understanding? (+reward)
  💎 Was it a non-obvious pick? (+reward)
  ❌ Was it a waste of time? (−reward)
        ↓
Results shown side-by-side with before/after toggle
```

---

#### 📊 Real calibration numbers (5 seeds × 17 researchers)

| Strategy | Mean reward | What it means |
|----------|------------|---------------|
| Random | −0.01 | Noise — proves reward is calibrated |
| Trending | +1.11 | Good, but not personalized |
| Dense Retrieval | +0.41 | Relevant, but obvious picks |
| **Blindspot (before RL)** | **−0.47** | Base model struggles |
| **Blindspot (after SFT)** | **+1.85** | RL learned what each person needs |
| Oracle (upper bound) | +2.77 | What perfect knowledge would score |

---

#### 🏗️ Architecture

- **Training:** Qwen3.5-9B + LoRA via Unsloth, trained with TRL's SFTTrainer (3 epochs × 40 traces, H100)
- **This demo:** Zero GPU — all trained-policy responses pre-cached in `data/demo_cache.json`
- **Data:** 17 real ML researchers, 1,168 concepts, 282 reading paths, 62 adoption pairs
- **Held-out test:** 4 researchers never seen during training

---

**Code:** [github.com/vasarlalikhilavinash/blindspot-env](https://github.com/vasarlalikhilavinash/blindspot-env)

**Trained adapter:** [huggingface.co/Vasarlaavinash/blindspot-sft-1.5b](https://huggingface.co/Vasarlaavinash/blindspot-sft-1.5b)
""")


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
