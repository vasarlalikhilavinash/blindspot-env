#!/usr/bin/env python3
"""Generate notebooks/03_demo.ipynb — judge-facing Gradio demo."""
import json
from pathlib import Path

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip().splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip().splitlines(keepends=True)})


md("""
# 🎯 Blindspot · Live Judge Demo

**Try it yourself.** This notebook hosts an interactive demo of Blindspot — the OpenEnv environment for
**unknown-unknowns discovery**.

> Paste a paragraph about your work, or pick one of our real users. Watch 5 different policies
> compete to surface what you should be tracking but aren't.

**What's compared, side by side:**

| Policy | What it does |
|---|---|
| Random | Picks 3 concepts at random |
| Trending | Picks the 3 most-mentioned concepts (popularity bait) |
| Dense Retrieval | Picks the 3 cosine-nearest concepts (you likely already know these) |
| **Blindspot RL** | Trained GRPO policy on Qwen-9B — learns to surface **novel-but-relevant-and-onboardable** concepts |
| GPT-4 (generic) | What ChatGPT would tell you if you asked "what should I learn?" |

Each surfaced concept comes with:
- A **5-paper reading path** (foundational → frontier)
- A **comprehension lift number** (judges measured how much *better* readers understand the concept after reading the path vs just abstracts)
- A **verdict** based on real held-out adoption ground truth
""")


md("## 1. Setup")

code("""
%%bash
pip install -q gradio openai numpy
git clone https://github.com/vasarlalikhilavinash/blindspot-env || (cd blindspot-env && git pull)
""")

code("""
import sys, os
sys.path.insert(0, 'blindspot-env')
from scripts.blindspot_demo import BlindspotDemo

# Optional: enable GPT-4 side-by-side comparison
# In Colab: Tools → Settings → Secrets, add OPENAI_API_KEY
try:
    from google.colab import userdata
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
except Exception:
    pass

OPENAI_KEY = os.environ.get('OPENAI_API_KEY', '')
print(f"OpenAI comparison: {'ENABLED' if OPENAI_KEY else 'disabled (set OPENAI_API_KEY in Colab secrets to enable)'}")
""")


md("## 2. Hook up the trained model (optional)")

code("""
# If you've already trained the GRPO model in 02_training.ipynb, plug it in here.
# Otherwise the demo runs with an honest 'pre-training placeholder' policy.

llm_generate = None
TRAINED_PATH = 'blindspot-env/training/checkpoints/grpo'

if os.path.isdir(TRAINED_PATH):
    print('✓ trained adapter found, loading...')
    from unsloth import FastLanguageModel
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name='unsloth/Qwen3.5-9B-bnb-4bit',
        max_seq_length=4096+128, load_in_4bit=True)
    model.load_adapter(TRAINED_PATH)
    FastLanguageModel.for_inference(model)

    def llm_generate(messages):
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors='pt', add_generation_prompt=True
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.0)
        return tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
    print('✓ trained policy hooked up')
else:
    print('⚠️ no trained adapter — demo will use kNN-informed proxy as placeholder')
""")


md("## 3. Build the demo engine")

code("""
def gpt_compare(paragraph: str) -> str:
    if not OPENAI_KEY:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    resp = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{
            'role': 'system',
            'content': 'You are a knowledgeable advisor. Given a description of someone\\'s work, '
                       'tell them in ONE short paragraph (max 80 words) what important AI/ML concepts '
                       'they should be tracking but probably aren\\'t. Be concrete.'
        }, {'role': 'user', 'content': paragraph}],
        max_tokens=200,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()

demo = BlindspotDemo(
    llm_generate=llm_generate,
    openai_compare=gpt_compare if OPENAI_KEY else None,
)
print(f'✓ demo engine ready: {len(demo.users)} users, {len(demo.catalog)} concepts')
""")


md("## 4. Pre-canned personas (one-click examples for judges)")

code("""
PERSONAS = {
    '🏥 Healthcare AI Lead':
        'I lead an applied AI team at a healthcare company. We build clinical decision-support agents '
        'using LLMs and RAG over electronic health records. Lately I worry about hallucination in '
        'long-form medical answers and whether our QA evaluations are even meaningful without expensive '
        'doctor reviews.',
    '💰 Fintech ML Engineer':
        'I am a senior ML engineer at a fintech, building retrieval-augmented LLM systems for '
        'compliance Q&A. I work with embeddings, vector search, and prompt engineering. I want to '
        'know what I am missing — agentic patterns? new fine-tuning tricks? something else?',
    '🧬 Bio-AI Founder':
        'I am a founder building protein-design LLMs for early-stage drug discovery. We fine-tune '
        'on assay data and use RAG over molecular databases. I am drowning in arxiv and feel like I '
        'am always 6 months behind on alignment / evaluation / safety methods.',
    '🎓 ML Researcher (real user from data)':
        'PICK_REAL_USER',  # special marker — UI will swap in real user picker
}
""")


md("## 5. Renderer (HTML output for Gradio)")

code('''
import html
def render_html(report):
    p = report['profile']
    out = []
    mode = p.get('mode', 'paragraph-match')
    out.append('<div style=\"font-family: -apple-system, sans-serif; max-width: 1100px;\">')

    # Profile banner
    badge = '🟢 REAL USER' if mode == 'real-user' else (
        '🟡 WEAK MATCH' if p.get('weak_match_warning') else '🔵 PARAGRAPH MATCH')
    out.append(f'<div style=\"background:#f0f4f8; padding:12px; border-radius:8px; margin-bottom:16px;\">')
    out.append(f'<b>{badge}</b> · matched user <code>{p[\"matched_user_id\"]}</code> '
               f'· cosine sim <b>{p[\"match_similarity\"]:.2f}</b> · pool size {p[\"candidate_pool_size\"]}<br>')
    if p.get('shared_keywords'):
        out.append(f'<small>shared keywords: {\", \".join(p[\"shared_keywords\"][:8])}</small><br>')
    out.append(f'<small style=\"color:#555;\">{html.escape(p[\"matched_summary\"][:200])}…</small>')
    out.append('</div>')

    # Reward bar comparison
    policies = report['policies']
    max_r = max(abs(v['reward']['total']) for v in policies.values()) or 1
    out.append('<h3>📊 Reward comparison (higher = better discovery skill)</h3>')
    out.append('<table style=\"width:100%; border-collapse:collapse;\">')
    palette = {'Random':'#888','Trending':'#cc7733','Dense Retrieval':'#3377cc',
               'Blindspot RL':'#22aa66','GPT-4':'#9966cc'}
    for name, res in policies.items():
        r = res['reward']['total']
        bar_w = int(abs(r)/max_r * 300)
        color = palette.get(name, '#666')
        sign = '' if r >= 0 else '-'
        align = 'left' if r >= 0 else 'right'
        out.append(f'<tr><td style=\"width:130px; padding:6px;\"><b>{name}</b></td>')
        out.append(f'<td style=\"width:80px; padding:6px;\">{r:+.2f}</td>')
        out.append(f'<td><div style=\"display:inline-block; width:{bar_w}px; height:18px; '
                   f'background:{color}; border-radius:3px;\"></div></td></tr>')
    out.append('</table>')

    # Per-policy expandables
    out.append('<h3 style=\"margin-top:24px;\">🔍 What each policy surfaced</h3>')
    for name, res in policies.items():
        r = res['reward']
        out.append(f'<details style=\"margin:8px 0; border-left:4px solid {palette.get(name,\"#666\")}; padding-left:12px;\">')
        out.append(f'<summary><b>{name}</b> — total {r[\"total\"]:+.2f} '
                   f'(adopt {r[\"adoption\"]:+.2f} · novel {r[\"novelty\"]:+.2f} · '
                   f'onboard {r[\"onboarding\"]:+.2f} · FP {r[\"false_positive\"]:+.2f})</summary>')
        out.append(f'<p><i>{res[\"description\"]}</i></p>')
        for card in res['cards']:
            color = '#22aa66' if card['adopted_by_user'] else '#cc3333'
            out.append(f'<div style=\"margin:8px 0; padding:10px; background:#fafafa; border-left:3px solid {color};\">')
            out.append(f'<b>[{card[\"concept_id\"]}] {html.escape(card[\"title\"])}</b><br>')
            out.append(f'<small>{html.escape(card[\"one_liner\"][:150])}</small><br>')
            tags = []
            if card['is_trending']: tags.append('🔥 trending')
            else: tags.append('💎 novel')
            if card['adopted_by_user']: tags.append('✅ user-adopted')
            else: tags.append('❌ no adoption signal')
            if card['comprehension_lift'] > 0:
                tags.append(f'📈 comp-lift {card[\"comprehension_lift\"]:+.2f}')
            out.append(' · '.join(tags))
            out.append(f'<br><b>verdict:</b> {card[\"verdict\"]}<br>')
            if card['reading_path']:
                out.append('<details><summary>📚 5-paper reading path (foundational → frontier)</summary><ol>')
                for paper in card['reading_path'][:5]:
                    out.append(f'<li>{html.escape(paper.get(\"title\",\"?\"))} ({paper.get(\"year\",\"?\")})</li>')
                out.append('</ol></details>')
            out.append('</div>')
        if res['meta'].get('reasoning'):
            out.append(f'<details><summary>💭 model reasoning</summary><pre style=\"white-space:pre-wrap;\">'
                       f'{html.escape(res[\"meta\"][\"reasoning\"])}</pre></details>')
        out.append('</details>')

    # ChatGPT comparison
    if report.get('chatgpt_baseline'):
        out.append('<h3 style=\"margin-top:24px;\">💬 What a generic LLM would tell you</h3>')
        out.append(f'<div style=\"background:#fff3e0; padding:12px; border-radius:6px;\">'
                   f'{html.escape(report[\"chatgpt_baseline\"])}</div>')
        out.append('<p><small>↑ Notice: this is generic and not grounded in YOUR concept catalog. '
                   'Blindspot picks specific concepts from a 1,168-item catalog with measured '
                   'adoption + comprehension data.</small></p>')

    out.append('</div>')
    return '\\n'.join(out)
''')


md("## 6. Gradio UI")

code('''
import gradio as gr

USER_OPTIONS = list(demo.users.keys())

def run_demo(persona_choice, paragraph, real_user_id, use_real_user):
    if use_real_user and real_user_id:
        report = demo.compare_all(user_id=real_user_id)
    elif persona_choice and persona_choice in PERSONAS and PERSONAS[persona_choice] != 'PICK_REAL_USER':
        # Pass the persona name as cache_key so trained-policy responses
        # are served from data/demo_cache.json (no GPU needed at inference time).
        slug = persona_choice.split(' ', 1)[-1].lower().replace(' ', '_')
        report = demo.compare_all(paragraph=PERSONAS[persona_choice], persona_key=slug)
    elif paragraph and paragraph.strip():
        report = demo.compare_all(paragraph=paragraph)
    else:
        return '<i>Pick a persona, paste a paragraph, or select a real user above.</i>'
    return render_html(report)

with gr.Blocks(title='Blindspot Demo', theme=gr.themes.Soft()) as ui:
    gr.Markdown('# 🎯 Blindspot — Unknown-Unknowns Discovery')
    gr.Markdown('Surface concepts you should be tracking, but currently aren\\'t.')

    with gr.Tabs():
        with gr.TabItem('🚀 Try a persona'):
            persona = gr.Radio(choices=[k for k in PERSONAS if PERSONAS[k] != 'PICK_REAL_USER'],
                               label='Pick a persona')
            paragraph = gr.Textbox(label='OR paste your own paragraph (LinkedIn bio, job description, etc.)',
                                   lines=4, placeholder='I work on...')
            run_btn1 = gr.Button('🔥 Run Blindspot', variant='primary')

        with gr.TabItem('🎓 Use a real user from our dataset'):
            gr.Markdown('These are 17 real ML researchers from Semantic Scholar. We have their '
                       'actual post-T adoption ground truth, so the rewards here are real measurements.')
            real_user = gr.Dropdown(choices=USER_OPTIONS, label='Pick a real user',
                                    value=USER_OPTIONS[0])
            run_btn2 = gr.Button('🔥 Run on real user', variant='primary')

    output = gr.HTML()

    run_btn1.click(fn=lambda p, par: run_demo(p, par, None, False),
                   inputs=[persona, paragraph], outputs=output)
    run_btn2.click(fn=lambda r: run_demo(None, None, r, True),
                   inputs=[real_user], outputs=output)

ui.launch(share=True, debug=False)
''')


md("""
## 7. (Optional) Headline metrics

After judges have played with the live demo, show them the aggregate proof:
the bar chart from the training notebook showing trained-policy reward vs all baselines.
""")

code("""
from IPython.display import Image, display
for img in ['blindspot-env/plots/comparison_with_trained.png',
            'blindspot-env/plots/decomposition_with_trained.png',
            'blindspot-env/plots/per_user_reward.png']:
    if os.path.exists(img):
        display(Image(img))
""")


# ─────────────────────────── write ───────────────────────────
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = Path('notebooks/03_demo.ipynb')
out.write_text(json.dumps(nb, indent=1))
print(f'wrote {out} with {len(CELLS)} cells')
