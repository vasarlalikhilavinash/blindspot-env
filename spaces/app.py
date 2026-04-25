"""Hugging Face Space entrypoint for Blindspot demo.

Architecture:
  - No GPU. Trained-policy responses are served from data/demo_cache.json
    (precomputed in Colab and committed to the repo).
  - Pre-training responses served from data/demo_cache_pretrain.json so the
    Before/After toggle shows the actual lift from GRPO training.
  - Baselines (Random/Trending/Dense) and the kNN proxy run on CPU instantly.
"""
from __future__ import annotations
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


# ──────────────────────────── HTML render ────────────────────────────
def render_html(report, focus="Blindspot RL"):
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
        kws = ', '.join(f"<code>{html.escape(k)}</code>" for k in p['shared_keywords'][:8])
        out.append(f"<small><b>matched on:</b> {kws}</small><br>")
    out.append(f"<small style='color:#555;'>{html.escape(p['matched_summary'][:240])}…</small></div>")

    policies = report['policies']

    # Headline before/after lift
    before_r = policies.get('Blindspot (pre-training)', {}).get('reward', {}).get('total', 0)
    after_r  = policies.get('Blindspot RL', {}).get('reward', {}).get('total', 0)
    lift = after_r - before_r
    out.append(f"<div style='background:#e8f5e9;padding:14px;border-radius:8px;"
               f"margin-bottom:16px;text-align:center;font-size:18px;'>"
               f"<b>📈 GRPO lift on this query:</b> "
               f"<span style='color:#aa4488;'>{before_r:+.2f}</span> "
               f"<b>→</b> <span style='color:#22aa66;'>{after_r:+.2f}</span> "
               f"<b style='color:{'#22aa66' if lift > 0 else '#cc3333'};'>"
               f"({'+' if lift >= 0 else ''}{lift:.2f})</b></div>")

    # Reward bar chart
    max_r = max(abs(v['reward']['total']) for v in policies.values()) or 1
    out.append("<h3>📊 Reward comparison (higher = better discovery skill)</h3>")
    out.append("<table style='width:100%;border-collapse:collapse;'>")
    for name, res in policies.items():
        r = res['reward']['total']
        bar_w = int(abs(r) / max_r * 300)
        color = PALETTE.get(name, '#666')
        latency = res.get('latency_ms', 0)
        out.append(f"<tr><td style='width:200px;padding:6px;'><b>{name}</b></td>"
                   f"<td style='width:80px;padding:6px;'>{r:+.2f}</td>"
                   f"<td style='width:340px;'><div style='display:inline-block;width:{bar_w}px;"
                   f"height:18px;background:{color};border-radius:3px;'></div></td>"
                   f"<td style='color:#888;font-size:11px;'>⚡ {latency:.1f}ms</td></tr>")
    out.append("</table>")

    out.append("<h3 style='margin-top:24px;'>🔍 What each policy surfaced</h3>")
    for name, res in policies.items():
        r = res['reward']
        cache_tag = ''
        if res['meta'].get('used_cache'):
            cache_tag = ' 💾 cached'
        elif res['meta'].get('used_trained_model'):
            cache_tag = ' 🤖 live-model'
        if res['meta'].get('used_nearest_neighbor'):
            cache_tag += f" 🔁 NN→{res['meta']['used_nearest_neighbor']}"

        is_open = (name == focus)
        out.append(f"<details {'open' if is_open else ''} "
                   f"style='margin:8px 0;border-left:4px solid {PALETTE.get(name,\"#666\")};"
                   f"padding-left:12px;'>")
        out.append(f"<summary><b>{name}</b>{cache_tag} — total {r['total']:+.2f} "
                   f"(adopt {r['adoption']:+.2f} · novel {r['novelty']:+.2f} · "
                   f"onboard {r['onboarding']:+.2f} · FP {r['false_positive']:+.2f})</summary>")
        out.append(f"<p><i>{html.escape(res['description'])}</i></p>")
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
                   "Blindspot picks specific concepts from a 1,168-item catalog with "
                   "measured adoption + comprehension data.</small></p>")

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
# 🎯 Blindspot — Unknown-Unknowns Discovery

Surface AI/ML concepts you should be tracking, but currently aren't.

Pick a **real user** (ground-truth rewards apply), a **persona** (cached trained-policy
responses), or paste a **paragraph**. Each surfaced concept comes with: a 5-paper
reading path, a measured comprehension lift, and a verdict against held-out adoption
ground truth.

📈 The headline number is the **GRPO lift**: pre-training Blindspot vs post-training
Blindspot, on the same query.
"""

PERSONA_LABELS = {
    "🤖 LLM Agents Researcher (frontier lab)": "llm_agents_researcher",
    "🔬 Diffusion Models PhD Student":         "diffusion_phd_student",
    "⚙️ ML Infra Engineer (50-person startup)": "ml_infra_engineer",
}

def run_real_user(uid, focus_label):
    focus = "Blindspot RL" if focus_label == "After (trained)" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(user_id=uid), focus=focus)

def run_persona(persona_label, focus_label):
    key = PERSONA_LABELS.get(persona_label)
    if not key:
        return "<i>Unknown persona</i>"
    focus = "Blindspot RL" if focus_label == "After (trained)" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(paragraph=PERSONAS[key], persona_key=key),
                       focus=focus)

def run_paragraph(text, focus_label):
    if not text or not text.strip():
        return "<i>Paste a paragraph above and click Run.</i>"
    focus = "Blindspot RL" if focus_label == "After (trained)" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(paragraph=text), focus=focus)


with gr.Blocks(title="Blindspot Demo", theme=gr.themes.Soft()) as ui:
    gr.Markdown(INTRO)
    with gr.Tabs():
        with gr.TabItem("🎓 Real user"):
            gr.Markdown("17 real ML researchers (13 train + **4 held-out test**). "
                        "Rewards are measured against held-out post-T adoption.")
            with gr.Row():
                real_user = gr.Dropdown(choices=USER_OPTIONS, label="Pick a user",
                                        value=USER_OPTIONS[0][1], scale=4)
                focus1 = gr.Radio(choices=["After (trained)", "Before (pre-training)"],
                                  value="After (trained)", label="Focus", scale=2)
            btn1 = gr.Button("🔥 Run", variant="primary")
            out1 = gr.HTML()
            btn1.click(run_real_user, inputs=[real_user, focus1], outputs=out1)

        with gr.TabItem("🚀 Persona"):
            gr.Markdown("Trained-policy responses for 3 ML-researcher personas (cached).")
            with gr.Row():
                persona = gr.Radio(choices=list(PERSONA_LABELS.keys()),
                                   value=list(PERSONA_LABELS.keys())[0],
                                   label="Pick a persona", scale=4)
                focus2 = gr.Radio(choices=["After (trained)", "Before (pre-training)"],
                                  value="After (trained)", label="Focus", scale=2)
            btn2 = gr.Button("🔥 Run", variant="primary")
            out2 = gr.HTML()
            btn2.click(run_persona, inputs=[persona, focus2], outputs=out2)

        with gr.TabItem("✍️ Your own paragraph"):
            gr.Markdown("Paste a LinkedIn bio / job description / a paragraph about your "
                        "work. Falls back to nearest-neighbor cached response (the trained "
                        "policy's output for the closest of 17 real users).")
            with gr.Row():
                text = gr.Textbox(lines=5, placeholder="I work on...", scale=4)
                focus3 = gr.Radio(choices=["After (trained)", "Before (pre-training)"],
                                  value="After (trained)", label="Focus", scale=2)
            btn3 = gr.Button("🔥 Run", variant="primary")
            out3 = gr.HTML()
            btn3.click(run_paragraph, inputs=[text, focus3], outputs=out3)

        with gr.TabItem("📚 Concept catalog"):
            gr.Markdown("Browse the full 1,168-concept catalog Blindspot surfaces from. "
                        "Each concept has reading paths, comprehension scores, and "
                        "user-adoption ground truth.")
            with gr.Row():
                ft = gr.Textbox(label="Filter by keyword", placeholder="e.g. diffusion")
                so = gr.Radio(choices=["all", "trending", "novel"], value="all",
                              label="Show only")
            btn4 = gr.Button("Search", variant="primary")
            out4 = gr.HTML(value=render_catalog())
            btn4.click(render_catalog, inputs=[ft, so], outputs=out4)

        with gr.TabItem("📖 About"):
            gr.Markdown("""
### How this demo works

- **Trained Blindspot policy** = Qwen3.5-9B + LoRA, fine-tuned with **GRPO** against
  the Blindspot reward (adoption + novelty + onboarding − false-positive).
- **Pre-training Blindspot** = same base model, **adapter disabled** — shows what the
  base model would have surfaced before GRPO. The Before/After delta is the lift from
  the Blindspot reward.
- **All trained-policy responses are deterministic (temp=0)** and cached in
  `data/demo_cache.json` and `data/demo_cache_pretrain.json` so this Space serves
  the real GRPO outputs with **zero GPU at request time**.
- **Code:** [github.com/vasarlalikhilavinash/blindspot-env](https://github.com/vasarlalikhilavinash/blindspot-env)
- **Trained adapter:** [huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo](https://huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo)
""")


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
