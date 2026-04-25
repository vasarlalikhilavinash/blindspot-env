#!/usr/bin/env python3
"""Generate notebooks/02_training.ipynb with rich pre/during/post training analysis."""
import json
from pathlib import Path

CELLS = []


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(keepends=True),
    })


def code(text: str):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(keepends=True),
    })


# ─────────────────────────── 0. Title ───────────────────────────
md("""
# Blindspot · GRPO Training & Analysis (Colab + A100)

End-to-end training notebook for the **Blindspot** OpenEnv. Trains
**Qwen3.5-9B (4-bit, LoRA via Unsloth)** with TRL's `GRPOTrainer`
against rewards computed by a live HTTP env server.

This notebook is split into 4 sections:

1. **Setup** — install deps, clone repo, boot env, sanity checks.
2. **Pre-training analysis** — explore the dataset (users, concepts,
   reading paths, comprehension lifts), then run *all* baselines and
   plot them so we have the "before" picture.
3. **Training** — load model, build prompts/reward, run GRPO, log live
   reward curves with running statistics.
4. **Post-training analysis** — evaluate the trained policy across all
   users with multiple seeds; plot trained-vs-baselines, per-component
   reward breakdown, per-user lift, action-type distribution, and
   qualitative sample episodes.

Estimated cost on A100: ~1.5–3 hours wall time, ~33 compute units.
""")


# ─────────────────────────── 1. Setup ───────────────────────────
md("""
---
## 1. Setup
""")

code("""
%%bash
pip install -q --upgrade unsloth trl 'openenv-core[core]' vllm matplotlib datasets peft accelerate bitsandbytes seaborn pandas
pip install -q --upgrade requests websocket-client
""")

code("""
%%bash
# Clone the project (idempotent) and boot the env server in the background.
git clone https://github.com/vasarlalikhilavinash/blindspot-env || (cd blindspot-env && git pull)
cd blindspot-env && python scripts/build_synthetic_seed.py 2>/dev/null || true
(cd blindspot-env && nohup uvicorn server.app:app --host 0.0.0.0 --port 8000 \\
        --log-level warning > /tmp/blindspot.log 2>&1 &)
sleep 6 && curl -s http://localhost:8000/state | head -c 300 || echo "(server may still be starting)"
""")

code("""
import sys, requests, json
sys.path.insert(0, 'blindspot-env')
ENV_URL = 'http://localhost:8000'

# Sanity check: env reachable, dataset shape OK
r = requests.post(f'{ENV_URL}/reset', json={}).json()
obs = r.get('observation', r) or {}
print(f"✓ env reachable")
print(f"  user pool size : {len(obs.get('user_id_pool', []))}")
print(f"  candidates    : {len(obs.get('candidate_concepts', []))}")
""")


# ─────────────────────── 2. Pre-training EDA ──────────────────────
md("""
---
## 2. Pre-training analysis

Inspect the data the env was built on. This is the same pre-computed
artifact set that powers the reward function during training.
""")

code("""
import json, statistics
from pathlib import Path

DATA = Path('blindspot-env/data')

users = json.load(open(DATA/'user_summaries.json'))
catalog = json.load(open(DATA/'concept_catalog.json'))
pool = json.load(open(DATA/'concept_pool_per_user.json'))
adoption = json.load(open(DATA/'ground_truth_adoption.json'))
comp = json.load(open(DATA/'comprehension_scores.json'))
paths = json.load(open(DATA/'reading_paths.json'))
nov = json.load(open(DATA/'novelty_flags.json'))

stats = {
    'n_users': len(users),
    'n_concepts_total': len(catalog),
    'n_concepts_in_pool': len({c for v in pool.values() for c in v}),
    'n_reading_paths': len(paths),
    'n_novel_flagged': sum(1 for v in nov.values() if v),
    'n_adoption_pairs': sum(len(v) for v in adoption.values()),
    'n_comprehension_pairs': sum(len(v) for v in comp.values()),
    'pool_size_per_user_mean': round(statistics.mean(len(v) for v in pool.values()), 1),
    'adoption_per_user_mean': round(statistics.mean(len(v) for v in adoption.values()), 1),
    'comprehension_lift_mean': round(statistics.mean(
        l for v in comp.values() for l in v.values()) or 0, 3) if any(v for v in comp.values()) else 0,
}
import pandas as pd
display(pd.DataFrame(list(stats.items()), columns=['stat', 'value']).style.hide(axis='index'))
""")

code("""
# Distribution plots: pool sizes, adoption per user, path lengths, comprehension lifts
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(11, 7))

ax = axes[0, 0]
sizes = [len(v) for v in pool.values()]
ax.hist(sizes, bins=10, color='#3377cc', edgecolor='white')
ax.set_title(f'Concept pool size per user (n={len(pool)} users)')
ax.set_xlabel('# concepts in pool'); ax.set_ylabel('# users')
ax.axvline(np.mean(sizes), color='red', ls='--', lw=1, label=f'mean={np.mean(sizes):.1f}')
ax.legend()

ax = axes[0, 1]
adopt_counts = [len(v) for v in adoption.values()]
ax.hist(adopt_counts, bins=range(0, max(adopt_counts)+2), color='#22aa66', edgecolor='white')
ax.set_title('Ground-truth adoptions per user')
ax.set_xlabel('# adopted concepts'); ax.set_ylabel('# users')
ax.axvline(np.mean(adopt_counts), color='red', ls='--', lw=1, label=f'mean={np.mean(adopt_counts):.1f}')
ax.legend()

ax = axes[1, 0]
plens = [len(p) for p in paths.values()]
ax.hist(plens, bins=range(1, max(plens)+2), color='#cc7733', edgecolor='white')
ax.set_title('Reading-path length distribution')
ax.set_xlabel('# papers in path'); ax.set_ylabel('# concepts')

ax = axes[1, 1]
lifts = [l for v in comp.values() for l in v.values()]
if lifts:
    ax.hist(lifts, bins=8, color='#9966cc', edgecolor='white')
    ax.set_title(f'Comprehension lift distribution (n={len(lifts)} pairs)')
    ax.set_xlabel('judge accuracy lift'); ax.set_ylabel('# (user, concept) pairs')
    ax.axvline(np.mean(lifts), color='red', ls='--', lw=1, label=f'mean={np.mean(lifts):.3f}')
    ax.legend()
else:
    ax.text(0.5, 0.5, 'no lift data', ha='center', va='center')
    ax.set_title('Comprehension lifts')

plt.tight_layout()
plt.savefig('blindspot-env/plots/data_distributions.png', dpi=120, bbox_inches='tight')
plt.show()
""")

code("""
# Run all baselines + oracle on the live env (these are the "before" numbers).
%cd blindspot-env
!python scripts/make_plots.py 2>&1 | tail -20
%cd ..

from IPython.display import Image, display
display(Image('blindspot-env/plots/baseline_comparison.png'))
display(Image('blindspot-env/plots/reward_decomposition.png'))

# Persist the pre-training summary so we can compare against trained policy
import json, shutil
shutil.copyfile('blindspot-env/plots/summary.json', 'blindspot-env/plots/summary_pretraining.json')
pretrain = json.load(open('blindspot-env/plots/summary_pretraining.json'))
print('\\nPre-training baseline rewards (mean ± std):')
for b in pretrain:
    print(f"  {b['name']:25s} {b['total']:+.3f} ± {b['std']:.2f}")
""")


# ─────────────────────────── 3. Training ───────────────────────────
md("""
---
## 3. Training

Load the base 9B model, attach LoRA adapters, build the prompt
dataset, and run GRPO with `num_generations=8` per prompt.
""")

code("""
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name='unsloth/Qwen3.5-9B-bnb-4bit',
    max_seq_length=4096+128,
    load_in_4bit=True,
)
import os
if os.path.isdir('blindspot-env/training/checkpoints/sft'):
    model.load_adapter('blindspot-env/training/checkpoints/sft')
    print('✓ attached SFT adapter')
FastLanguageModel.for_training(model)

# Quick GPU memory snapshot
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f'GPU: {torch.cuda.get_device_name(0)} | free {free/1e9:.1f}/{total/1e9:.1f} GB')
""")

code("""
# Build prompt dataset by sampling (user, seed) pairs from the env.
import sys; sys.path.insert(0, 'blindspot-env')
from training.grpo_train import SYSTEM_PROMPT, render_obs, parse_action
import requests, random
from datasets import Dataset

ENV_URL = 'http://localhost:8000'
r0 = requests.post(f'{ENV_URL}/reset', json={}).json()
user_pool = (r0.get('observation', r0) or {}).get('user_id_pool', [])
rng = random.Random(0)
rows = []
for _ in range(256):
    uid = rng.choice(user_pool); seed = rng.randrange(1_000_000)
    obs = requests.post(f'{ENV_URL}/reset', json={'user_id': uid, 'seed': seed}).json()
    obs = obs.get('observation', obs)
    rows.append({'prompt': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': render_obs(obs)},
    ], 'user_id': uid, 'seed': seed})
ds = Dataset.from_list(rows)
print(f'Built {len(ds)} prompts across {len(set(r["user_id"] for r in rows))} unique users')
""")

code("""
# Reward function: batch-call the env and return per-completion totals.
# Also stash the breakdowns so we can chart per-component reward later.
import requests
REWARD_LOG = []  # (step_idx, total, breakdown)

def reward_fn(prompts, completions, user_id=None, seed=None, **kw):
    out = []
    uids  = user_id if isinstance(user_id, list) else [user_id]*len(completions)
    seeds = seed    if isinstance(seed, list)    else [seed]*len(completions)
    for prompt_msgs, completion, uid, sd in zip(prompts, completions, uids, seeds):
        text = completion if isinstance(completion, str) else completion[-1].get('content', '')
        action = parse_action(text)
        if not action or 'type' not in action:
            out.append(-0.05); REWARD_LOG.append((-0.05, {})); continue
        try:
            payload = {}
            if uid is not None: payload['user_id'] = uid
            if sd  is not None: payload['seed']    = sd
            requests.post(f'{ENV_URL}/reset', json=payload).raise_for_status()
            r = requests.post(f'{ENV_URL}/step', json={'action': action}).json()
            rs = requests.post(f'{ENV_URL}/step', json={'action': {'type': 'stop'}}).json()
            br = (rs.get('observation', rs) or {}).get('reward_breakdown') or {}
            tot = float(br.get('total', r.get('reward', 0.0) or 0.0))
            out.append(tot); REWARD_LOG.append((tot, br))
        except Exception:
            out.append(-0.1); REWARD_LOG.append((-0.1, {}))
    return out
""")

code("""
from trl import GRPOConfig, GRPOTrainer
cfg = GRPOConfig(
    output_dir='blindspot-env/training/checkpoints/grpo',
    learning_rate=5e-6,
    max_steps=400,
    num_generations=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=4096,
    max_completion_length=64,
    logging_steps=5,
    save_steps=100,
    bf16=True,
    report_to='none',
)
trainer = GRPOTrainer(model=model, processing_class=tokenizer,
                      reward_funcs=[reward_fn], args=cfg, train_dataset=ds)

print(f'Starting GRPO: {cfg.max_steps} steps × {cfg.num_generations} rollouts/step '
      f'= {cfg.max_steps*cfg.num_generations} reward queries')
trainer.train()
trainer.save_model('blindspot-env/training/checkpoints/grpo')
print('✓ training complete, adapter saved')
""")


# ─────────────────────── 4. Post-training analysis ──────────────────────
md("""
---
## 4. Post-training analysis

Charts below quantify how much GRPO improved over the baselines.
""")

code("""
# 4.1 Live reward curve from training (running mean over rollouts).
import numpy as np
import matplotlib.pyplot as plt

totals = np.array([t for t, _ in REWARD_LOG])
window = max(1, len(totals)//40)
running = np.convolve(totals, np.ones(window)/window, mode='valid')

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(totals, alpha=0.25, color='#888', label='per-rollout reward')
ax.plot(np.arange(len(running))+window-1, running, color='#22aa66', lw=2, label=f'running mean (k={window})')
ax.axhline(0, color='black', ls='--', lw=0.7)
ax.set_xlabel('rollout #'); ax.set_ylabel('reward')
ax.set_title(f'GRPO training reward — {len(totals)} rollouts')
ax.legend()
plt.tight_layout()
plt.savefig('blindspot-env/plots/training_reward_curve.png', dpi=120, bbox_inches='tight')
plt.show()

print(f'first 10% mean reward : {totals[:len(totals)//10].mean():+.3f}')
print(f'last  10% mean reward : {totals[-len(totals)//10:].mean():+.3f}')
print(f'gain                  : {totals[-len(totals)//10:].mean() - totals[:len(totals)//10].mean():+.3f}')
""")

code("""
# 4.2 Per-component reward trajectory (adoption / novelty / onboarding / ...).
import pandas as pd
comp_rows = []
for i, (tot, br) in enumerate(REWARD_LOG):
    if not br: continue
    comp_rows.append({'step': i, **br})
if comp_rows:
    df = pd.DataFrame(comp_rows)
    cols = [c for c in ['adoption','novelty','onboarding','efficiency','false_positive'] if c in df.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = max(1, len(df)//40)
    for c, color in zip(cols, ['#22aa66','#3377cc','#9966cc','#888888','#cc3333']):
        smooth = df[c].rolling(w, min_periods=1).mean()
        ax.plot(df['step'], smooth, label=c.replace('_',' '), color=color, lw=1.6)
    ax.axhline(0, color='black', ls='--', lw=0.7)
    ax.legend(loc='best', ncol=5, frameon=False)
    ax.set_xlabel('rollout #'); ax.set_ylabel('component reward (running mean)')
    ax.set_title('Reward decomposition over training')
    plt.tight_layout()
    plt.savefig('blindspot-env/plots/training_component_curves.png', dpi=120, bbox_inches='tight')
    plt.show()
""")

code("""
# 4.3 Evaluate trained policy on every user, multiple seeds.
import torch, requests, statistics, json
from training.grpo_train import SYSTEM_PROMPT, render_obs, parse_action
FastLanguageModel.for_inference(model)

def trained_episode(user_id, seed):
    obs = requests.post(f'{ENV_URL}/reset', json={'user_id': user_id, 'seed': seed}).json()
    obs = obs.get('observation', obs)
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': render_obs(obs)},
    ]
    inputs = tokenizer.apply_chat_template(msgs, return_tensors='pt', add_generation_prompt=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.0)
    completion = tokenizer.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
    action = parse_action(completion) or {'type': 'stop'}
    requests.post(f'{ENV_URL}/reset', json={'user_id': user_id, 'seed': seed})
    requests.post(f'{ENV_URL}/step', json={'action': action})
    rs = requests.post(f'{ENV_URL}/step', json={'action': {'type':'stop'}}).json()
    br = (rs.get('observation', rs) or {}).get('reward_breakdown') or {}
    return br, action, completion

results = []
for uid in user_pool:
    for s in range(5):
        br, act, comp = trained_episode(uid, s)
        results.append({'user': uid, 'seed': s, 'action_type': act.get('type'), 'completion': comp, **br})

import pandas as pd
df_trained = pd.DataFrame(results)
print(f'Evaluated {len(df_trained)} episodes')
print(f'  trained mean reward : {df_trained["total"].mean():+.3f} ± {df_trained["total"].std():.2f}')
""")

code("""
# 4.4 Trained policy vs baselines vs oracle — bar chart.
import json, matplotlib.pyplot as plt
pretrain = json.load(open('blindspot-env/plots/summary_pretraining.json'))
all_rows = list(pretrain) + [{
    'name': 'GRPO (trained)',
    'total': float(df_trained['total'].mean()),
    'std': float(df_trained['total'].std()),
    'adoption': float(df_trained.get('adoption', pd.Series([0])).mean()),
    'novelty': float(df_trained.get('novelty', pd.Series([0])).mean()),
    'onboarding': float(df_trained.get('onboarding', pd.Series([0])).mean()),
    'efficiency': float(df_trained.get('efficiency', pd.Series([0])).mean()),
    'false_positive': float(df_trained.get('false_positive', pd.Series([0])).mean()),
}]
# Reorder: baselines then trained then oracle
ordered = ['Random','Trending','Dense Retrieval','GRPO (trained)','Oracle (upper bound)']
all_rows.sort(key=lambda r: ordered.index(r['name']) if r['name'] in ordered else 99)

fig, ax = plt.subplots(figsize=(8, 4.8))
names  = [r['name'] for r in all_rows]
totals = [r['total'] for r in all_rows]
stds   = [r['std'] for r in all_rows]
colors = ['#888888','#cc7733','#3377cc','#cc3377','#22aa66']
bars = ax.bar(names, totals, yerr=stds, capsize=6, color=colors[:len(names)])
for i, t in enumerate(totals):
    ax.text(i, t + 0.15, f'{t:+.2f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0, color='black', lw=0.7, ls='--')
ax.set_ylabel('Mean episode reward')
ax.set_title('Blindspot: trained policy vs baselines (real data)')
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig('blindspot-env/plots/comparison_with_trained.png', dpi=130, bbox_inches='tight')
plt.show()

# Save updated summary
with open('blindspot-env/plots/summary_with_trained.json','w') as f:
    json.dump(all_rows, f, indent=2)
""")

code("""
# 4.5 Reward decomposition stacked bar — what does GRPO win on?
import matplotlib.pyplot as plt, numpy as np
components = ['adoption','novelty','onboarding','efficiency','false_positive']
palette = {'adoption':'#22aa66','novelty':'#3377cc','onboarding':'#9966cc',
           'efficiency':'#888888','false_positive':'#cc3333'}

fig, ax = plt.subplots(figsize=(9, 5))
bottom_pos = np.zeros(len(all_rows)); bottom_neg = np.zeros(len(all_rows))
for c in components:
    vals = np.array([r.get(c, 0.0) for r in all_rows])
    bot = np.where(vals>=0, bottom_pos, bottom_neg)
    ax.bar(names, vals, bottom=bot, color=palette[c], label=c.replace('_',' '))
    bottom_pos += np.where(vals>=0, vals, 0)
    bottom_neg += np.where(vals< 0, vals, 0)
ax.axhline(0, color='black', lw=0.6)
ax.set_ylabel('Reward (signed components)')
ax.set_title('Reward decomposition: what each policy earns vs loses')
ax.legend(ncol=5, frameon=False, loc='upper left', bbox_to_anchor=(0,-0.07))
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig('blindspot-env/plots/decomposition_with_trained.png', dpi=130, bbox_inches='tight')
plt.show()
""")

code("""
# 4.6 Per-user reward — does training help all users equally?
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(11, 4.5))
agg = df_trained.groupby('user')['total'].agg(['mean','std']).sort_values('mean')
ax.bar(range(len(agg)), agg['mean'], yerr=agg['std'], capsize=4,
       color=['#cc3333' if x<0 else '#22aa66' for x in agg['mean']])
ax.axhline(0, color='black', lw=0.7, ls='--')
ax.set_xticks(range(len(agg))); ax.set_xticklabels([str(u)[:6] for u in agg.index], rotation=45, fontsize=8)
ax.set_ylabel('mean episode reward'); ax.set_xlabel('user (truncated id)')
ax.set_title('GRPO trained policy — per-user reward (5 seeds each)')
plt.tight_layout()
plt.savefig('blindspot-env/plots/per_user_reward.png', dpi=120, bbox_inches='tight')
plt.show()
print(f'  users with positive mean : {(agg["mean"]>0).sum()}/{len(agg)}')
""")

code("""
# 4.7 Action-type distribution — what's the trained policy actually doing?
import matplotlib.pyplot as plt
counts = df_trained['action_type'].fillna('(unparsable)').value_counts()
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(counts.index, counts.values, color=['#22aa66','#3377cc','#cc7733','#888888'][:len(counts)])
for i, v in enumerate(counts.values):
    ax.text(i, v+0.5, str(v), ha='center')
ax.set_ylabel('# episodes'); ax.set_title('Trained policy: action type distribution')
plt.tight_layout()
plt.savefig('blindspot-env/plots/action_type_distribution.png', dpi=120, bbox_inches='tight')
plt.show()
""")

code("""
# 4.8 Sample qualitative episodes — show a few full prompt → action → reward cases.
import textwrap, pandas as pd
sample = df_trained.sort_values('total', ascending=False).head(3)
for _, row in sample.iterrows():
    print('='*70)
    print(f'user={row["user"]}  seed={row["seed"]}  reward={row["total"]:+.2f}  action={row["action_type"]}')
    print('completion:')
    print(textwrap.fill(row['completion'][:300], width=70, initial_indent='  ', subsequent_indent='  '))
print('\\n— and the worst three —')
for _, row in df_trained.sort_values('total').head(3).iterrows():
    print('='*70)
    print(f'user={row["user"]}  seed={row["seed"]}  reward={row["total"]:+.2f}  action={row["action_type"]}')
    print(textwrap.fill(row['completion'][:300], width=70, initial_indent='  ', subsequent_indent='  '))
""")

code("""
# 4.9 Headline summary table.
import pandas as pd
rows = []
for r in all_rows:
    rows.append({
        'policy': r['name'],
        'mean_reward': round(r['total'], 3),
        'std': round(r['std'], 3),
        'adoption': round(r.get('adoption', 0), 2),
        'novelty': round(r.get('novelty', 0), 2),
        'onboarding': round(r.get('onboarding', 0), 2),
        'false_positive': round(r.get('false_positive', 0), 2),
    })
summary_df = pd.DataFrame(rows)

# % of oracle gap closed by trained policy (vs best baseline)
import json
trained_r = next(r for r in all_rows if r['name']=='GRPO (trained)')['total']
oracle_r  = next(r for r in all_rows if r['name'].startswith('Oracle'))['total']
best_baseline_r = max(r['total'] for r in all_rows
                      if r['name'] in ('Random','Trending','Dense Retrieval'))
gap_total  = oracle_r - best_baseline_r
gap_closed = trained_r - best_baseline_r
pct = (gap_closed / gap_total * 100) if gap_total > 0 else 0

print('\\nTRAINED-POLICY HEADLINE')
print('='*42)
print(f'  best baseline reward : {best_baseline_r:+.3f}')
print(f'  trained reward       : {trained_r:+.3f}')
print(f'  oracle reward        : {oracle_r:+.3f}')
print(f'  gap closed           : {pct:.1f}% of (oracle − best-baseline)')
display(summary_df)
""")


# ─────────────────────────── 5. Deploy ───────────────────────────
md("""
---
## 5. 🚀 Deploy: push adapter, precompute cache, deploy HF Space

This section makes the demo **available 24/7 for the 3-week judging window** with
**zero GPU at request time** by:

1. Pushing the trained LoRA adapter to the HF Hub.
2. Running the trained policy on all 17 users + 3 personas to build `data/demo_cache.json`.
3. Committing the cache back to GitHub.
4. Deploying a Gradio Space that reads from the cache (free CPU tier — runs forever).

**Required Colab secrets** (Colab → 🔑 → enable for this notebook):
- `HF_TOKEN` — write-scope token from https://huggingface.co/settings/tokens
- `GITHUB_TOKEN` — fine-grained token with repo write to `vasarlalikhilavinash/blindspot-env`
- `OPENAI_API_KEY` *(optional)* — for the GPT-4 baseline column on the live Space
""")

code("""
# Save trained adapter to disk first
model.save_pretrained('blindspot-env/training/checkpoints/grpo')
tokenizer.save_pretrained('blindspot-env/training/checkpoints/grpo')
print('✓ adapter saved to blindspot-env/training/checkpoints/grpo')
""")

code("""
# Pull secrets from Colab
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
try:
    os.environ['GITHUB_TOKEN'] = userdata.get('GITHUB_TOKEN')
except Exception:
    print('⚠️ no GITHUB_TOKEN secret — skip git-push step')
try:
    os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
except Exception:
    pass
print('✓ secrets loaded')
""")

code("""
# 5.1 — push LoRA adapter to HF Hub
%cd blindspot-env
!python scripts/push_to_hub.py
""")

code("""
# 5.2 — precompute trained-policy cache (17 users + 3 personas, ~3 min on A100)
!python scripts/precompute_demo_cache.py
""")

code("""
# 5.3 — commit cache back to GitHub so the Space can pick it up
import os, subprocess
tok = os.environ.get('GITHUB_TOKEN', '')
if tok:
    subprocess.run(['git', 'config', 'user.email', 'colab@blindspot.ai'], check=True)
    subprocess.run(['git', 'config', 'user.name',  'Blindspot Colab'], check=True)
    subprocess.run(['git', 'add', 'data/demo_cache.json'], check=True)
    r = subprocess.run(['git', 'commit', '-m', 'Phase D: trained-policy cache'])
    if r.returncode == 0:
        url = f'https://{tok}@github.com/vasarlalikhilavinash/blindspot-env.git'
        subprocess.run(['git', 'push', url, 'HEAD:main'], check=True)
        print('✓ cache committed and pushed')
    else:
        print('(nothing to commit)')
else:
    print('⚠️ no GITHUB_TOKEN — cache stays local; download data/demo_cache.json manually')
""")

code("""
# 5.4 — deploy the Gradio Space (free CPU tier; serves cached responses 24/7)
!python scripts/deploy_to_space.py
print()
print('🎉 Demo URL: https://huggingface.co/spaces/vasarlalikhilavinash/blindspot-demo')
print('   First build takes ~2 min; afterwards it stays warm.')
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

out = Path('notebooks/02_training.ipynb')
out.write_text(json.dumps(nb, indent=1))
print(f'wrote {out} with {len(CELLS)} cells')
