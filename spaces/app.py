"""Hugging Face Space entrypoint for Blindspot demo.

Architecture:
  - No GPU. Trained-policy responses are served from data/demo_cache.json
    (precomputed in Colab and committed to the repo).
  - Baselines (Random/Trending/Dense) and the kNN proxy run on CPU instantly.
  - Optional: if HF_INFERENCE_ENDPOINT + HF_TOKEN are set, ad-hoc paragraphs
    fall back to a hosted Inference Endpoint. Otherwise they use the proxy.

Deploy as a Gradio Space:
  - Hardware: CPU basic (free)
  - SDK: gradio
  - Public URL persists for 3+ weeks.
"""
from __future__ import annotations
import html
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.blindspot_demo import BlindspotDemo
from scripts.precompute_demo_cache import PERSONAS  # single source of truth

import gradio as gr


# ──────────────────────────── optional remote LLM ────────────────────────────
HF_ENDPOINT = os.environ.get("HF_INFERENCE_ENDPOINT", "").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

def remote_llm_generate(messages):
    """Optional bridge to a hosted HF Inference Endpoint serving the trained adapter.

    Only used for ad-hoc paragraphs (the cache covers users + personas).
    """
    if not HF_ENDPOINT or not HF_TOKEN:
        raise RuntimeError("no remote endpoint configured")
    import requests
    r = requests.post(
        f"{HF_ENDPOINT}/v1/chat/completions",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "model": "tgi",
            "messages": messages,
            "max_tokens": 64,
            "temperature": 0.0,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# Optional GPT-4 generic-comparison side panel (uses OPENAI_API_KEY secret if present)
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
                "content": "You are a knowledgeable advisor. Given a description of someone's "
                           "work, tell them in ONE short paragraph (max 80 words) what important "
                           "AI/ML concepts they should be tracking but probably aren't."
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
USER_OPTIONS = [(f"{uid}: {demo_engine.users[uid][:80]}…", uid) for uid in demo_engine.users.keys()]


# ──────────────────────────── HTML render ────────────────────────────
def render_html(report):
    p = report['profile']
    out = ["<div style='font-family:-apple-system,sans-serif;max-width:1100px;'>"]

    mode = p.get('mode', 'paragraph-match')
    badge = {'real-user': '🟢 REAL USER (ground-truth applies)',
             'persona': '🟣 PERSONA (cached trained-policy response)',
             'paragraph-match': '🔵 AD-HOC PARAGRAPH'}.get(mode, mode)
    if p.get('weak_match_warning'):
        badge += ' · ⚠️ weak match — consider picking a real user instead'

    out.append("<div style='background:#f0f4f8;padding:12px;border-radius:8px;margin-bottom:16px;'>")
    out.append(f"<b>{badge}</b> · matched user <code>{p['matched_user_id']}</code> · "
               f"cosine sim <b>{p['match_similarity']:.2f}</b> · "
               f"pool size {p['candidate_pool_size']}<br>")
    if p.get('shared_keywords'):
        out.append(f"<small>shared keywords: {', '.join(p['shared_keywords'][:8])}</small><br>")
    out.append(f"<small style='color:#555;'>{html.escape(p['matched_summary'][:240])}…</small></div>")

    policies = report['policies']
    max_r = max(abs(v['reward']['total']) for v in policies.values()) or 1
    out.append("<h3>📊 Reward comparison (higher = better discovery skill)</h3>")
    out.append("<table style='width:100%;border-collapse:collapse;'>")
    palette = {'Random': '#888', 'Trending': '#cc7733', 'Dense Retrieval': '#3377cc',
               'Blindspot RL': '#22aa66'}
    for name, res in policies.items():
        r = res['reward']['total']
        bar_w = int(abs(r) / max_r * 300)
        color = palette.get(name, '#666')
        out.append(f"<tr><td style='width:130px;padding:6px;'><b>{name}</b></td>"
                   f"<td style='width:80px;padding:6px;'>{r:+.2f}</td>"
                   f"<td><div style='display:inline-block;width:{bar_w}px;height:18px;"
                   f"background:{color};border-radius:3px;'></div></td></tr>")
    out.append("</table>")

    out.append("<h3 style='margin-top:24px;'>🔍 What each policy surfaced</h3>")
    for name, res in policies.items():
        r = res['reward']
        cache_tag = ' 💾 cached' if res['meta'].get('used_cache') else (
            ' 🤖 live-model' if res['meta'].get('used_trained_model') else '')
        out.append(f"<details {'open' if name == 'Blindspot RL' else ''} "
                   f"style='margin:8px 0;border-left:4px solid {palette.get(name,\"#666\")};"
                   f"padding-left:12px;'>")
        out.append(f"<summary><b>{name}</b>{cache_tag} — total {r['total']:+.2f} "
                   f"(adopt {r['adoption']:+.2f} · novel {r['novelty']:+.2f} · "
                   f"onboard {r['onboarding']:+.2f} · FP {r['false_positive']:+.2f})</summary>")
        out.append(f"<p><i>{res['description']}</i></p>")
        for card in res['cards']:
            color = '#22aa66' if card['adopted_by_user'] else '#cc3333'
            out.append(f"<div style='margin:8px 0;padding:10px;background:#fafafa;"
                       f"border-left:3px solid {color};'>")
            out.append(f"<b>[{card['concept_id']}] {html.escape(card['title'])}</b><br>"
                       f"<small>{html.escape(card['one_liner'][:160])}</small><br>")
            tags = []
            tags.append('🔥 trending' if card['is_trending'] else '💎 novel')
            tags.append('✅ user-adopted' if card['adopted_by_user'] else '❌ no adoption signal')
            if card['comprehension_lift'] > 0:
                tags.append(f"📈 comp-lift {card['comprehension_lift']:+.2f}")
            out.append(' · '.join(tags))
            out.append(f"<br><b>verdict:</b> {card['verdict']}<br>")
            if card['reading_path']:
                out.append("<details><summary>📚 5-paper reading path</summary><ol>")
                for paper in card['reading_path'][:5]:
                    out.append(f"<li>{html.escape(paper.get('title','?'))} "
                               f"({paper.get('year','?')})</li>")
                out.append("</ol></details>")
            out.append("</div>")
        if res['meta'].get('reasoning'):
            out.append(f"<details><summary>💭 model reasoning</summary>"
                       f"<pre style='white-space:pre-wrap;'>"
                       f"{html.escape(res['meta']['reasoning'])}</pre></details>")
        out.append("</details>")

    if report.get('chatgpt_baseline'):
        out.append("<h3 style='margin-top:24px;'>💬 What a generic LLM would tell you</h3>")
        out.append(f"<div style='background:#fff3e0;padding:12px;border-radius:6px;'>"
                   f"{html.escape(report['chatgpt_baseline'])}</div>")
        out.append("<p><small>↑ Generic, not grounded in YOUR concept catalog. "
                   "Blindspot picks specific concepts from a 1,168-item catalog with measured "
                   "adoption + comprehension data.</small></p>")

    out.append("</div>")
    return '\n'.join(out)


# ──────────────────────────── Gradio app ────────────────────────────
INTRO = """
# 🎯 Blindspot — Unknown-Unknowns Discovery

Surface concepts you should be tracking, but currently aren't.

Pick a **real user** (ground-truth rewards apply), a **persona** (cached trained-policy responses),
or paste a **paragraph** about your work.

Each surfaced concept comes with: a 5-paper reading path, a measured comprehension lift, and a
verdict against held-out adoption ground truth.
"""

def run_real_user(uid):
    return render_html(demo_engine.compare_all(user_id=uid))

def run_persona(persona_label):
    # Map label back to key
    label2key = {f"🏥 Healthcare AI Lead": "healthcare_ai_lead",
                 f"💰 Fintech ML Engineer": "fintech_ml_engineer",
                 f"🧬 Bio-AI Founder": "bio_ai_founder"}
    key = label2key.get(persona_label)
    if not key:
        return "<i>Unknown persona</i>"
    return render_html(demo_engine.compare_all(paragraph=PERSONAS[key], persona_key=key))

def run_paragraph(text):
    if not text or not text.strip():
        return "<i>Paste a paragraph above and click Run.</i>"
    return render_html(demo_engine.compare_all(paragraph=text))


with gr.Blocks(title="Blindspot Demo", theme=gr.themes.Soft()) as ui:
    gr.Markdown(INTRO)
    with gr.Tabs():
        with gr.TabItem("🎓 Real user"):
            gr.Markdown("17 real ML researchers from Semantic Scholar. Rewards here are "
                        "measured against held-out post-T adoption.")
            real_user = gr.Dropdown(choices=USER_OPTIONS, label="Pick a user",
                                    value=USER_OPTIONS[0][1])
            btn1 = gr.Button("🔥 Run", variant="primary")
            out1 = gr.HTML()
            btn1.click(run_real_user, inputs=real_user, outputs=out1)

        with gr.TabItem("🚀 Persona"):
            gr.Markdown("Trained-policy responses for 3 hand-crafted personas (cached).")
            persona = gr.Radio(
                choices=["🏥 Healthcare AI Lead", "💰 Fintech ML Engineer", "🧬 Bio-AI Founder"],
                label="Pick a persona", value="🏥 Healthcare AI Lead")
            btn2 = gr.Button("🔥 Run", variant="primary")
            out2 = gr.HTML()
            btn2.click(run_persona, inputs=persona, outputs=out2)

        with gr.TabItem("✍️ Your own paragraph"):
            gr.Markdown("Paste a LinkedIn bio / job description / a paragraph about your work. "
                        "Note: ad-hoc paragraphs use the kNN proxy unless a remote inference "
                        "endpoint is configured.")
            text = gr.Textbox(lines=5, placeholder="I work on...")
            btn3 = gr.Button("🔥 Run", variant="primary")
            out3 = gr.HTML()
            btn3.click(run_paragraph, inputs=text, outputs=out3)


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
