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
               f"<div style='font-size:13px;color:#555;margin-bottom:6px;'>Did GRPO training help for this researcher?</div>"
               f"<div style='font-size:20px;'>"
               f"Base model (before RL): <span style='color:#aa4488;font-weight:700;'>{before_r:+.2f}</span>"
               f" &nbsp;→&nbsp; "
               f"After GRPO training: <span style='color:#22aa66;font-weight:700;'>{after_r:+.2f}</span>"
               f" &nbsp; {lift_emoji} <span style='color:{lift_color};font-weight:700;'>{'+' if lift>=0 else ''}{lift:.2f}</span>"
               f"</div>"
               f"<div style='font-size:12px;color:#666;margin-top:6px;'>{lift_story}</div>"
               f"</div>")

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

Blindspot is an AI trained with **reinforcement learning (GRPO)** to act like a research assistant that:

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
- **The key comparison:** AI before training vs AI after GRPO training

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
    focus = "Blindspot RL" if focus_label == "After GRPO training ✅" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(user_id=uid), focus=focus)

def run_persona(persona_label, focus_label):
    key = PERSONA_LABELS.get(persona_label)
    if not key:
        return "<i>Unknown persona</i>"
    focus = "Blindspot RL" if focus_label == "After GRPO training ✅" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(paragraph=PERSONAS[key], persona_key=key),
                       focus=focus)

def run_paragraph(text, focus_label):
    if not text or not text.strip():
        return "<i>Paste a paragraph about your work and click Run.</i>"
    focus = "Blindspot RL" if focus_label == "After GRPO training ✅" else "Blindspot (pre-training)"
    return render_html(demo_engine.compare_all(paragraph=text), focus=focus)


with gr.Blocks(title="Blindspot — Unknown-Unknowns Discovery", theme=gr.themes.Soft()) as ui:
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
                    choices=["After GRPO training ✅", "Before training (base model)"],
                    value="After GRPO training ✅",
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
                    choices=["After GRPO training ✅", "Before training (base model)"],
                    value="After GRPO training ✅",
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
                    choices=["After GRPO training ✅", "Before training (base model)"],
                    value="After GRPO training ✅",
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
  Pre-training  → Qwen3.5-9B base model picks (no RL)
  GRPO trained  → Qwen3.5-9B + LoRA after 3,200 RL episodes ← this is Blindspot
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
| **Blindspot (after GRPO)** | **+1.85** | RL learned what each person needs |
| Oracle (upper bound) | +2.77 | What perfect knowledge would score |

---

#### 🏗️ Architecture

- **Training:** Qwen3.5-9B + LoRA via Unsloth, trained with TRL's GRPOTrainer (400 steps × 8 rollouts, A100)
- **This demo:** Zero GPU — all trained-policy responses pre-cached in `data/demo_cache.json`
- **Data:** 17 real ML researchers, 1,168 concepts, 282 reading paths, 62 adoption pairs
- **Held-out test:** 4 researchers never seen during training

---

**Code:** [github.com/vasarlalikhilavinash/blindspot-env](https://github.com/vasarlalikhilavinash/blindspot-env)

**Trained adapter:** [huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo](https://huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo)
""")


if __name__ == "__main__":
    ui.launch(server_name="0.0.0.0", server_port=7860)
