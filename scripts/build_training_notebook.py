#!/usr/bin/env python3
"""Generate notebooks/02_training.ipynb for Blindspot."""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

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


md(
    """
# Blindspot · Training, Evaluation, and Deployment

This Colab notebook is the end-to-end training path for Blindspot:

1. audit the real dataset and held-out split
2. run real-data baseline calibration
3. generate a small oracle SFT warm-start on the 13 training users
4. run GRPO on top of the SFT adapter
5. evaluate on held-out users
6. precompute both demo caches and deploy the CPU-only Hugging Face Space

The intended hardware target is a single A100 Colab runtime.
"""
)

md("## 1. Setup")

code(
    """
%%bash
set -e
python -m pip uninstall -y unsloth unsloth_zoo transformers trl datasets torchcodec >/dev/null 2>&1 || true
python -m pip install -q --upgrade pip wheel setuptools packaging
python -m pip install -q --upgrade --no-cache-dir \
  'openenv-core[core]' \
  'transformers==5.5.0' \
  'trl==0.24.0' \
  'datasets==4.3.0' \
  'numpy<2.1' \
  'torchao>=0.16.0' \
  peft accelerate bitsandbytes
python -m pip install -q --upgrade --no-cache-dir --no-deps \
  'unsloth_zoo==2026.4.9' \
  'unsloth==2026.4.8'
python - <<'PY'
import torch, transformers, trl, datasets
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print(f'torch={torch.__version__} cuda={torch.version.cuda}')
print(f'transformers={transformers.__version__} trl={trl.__version__} datasets={datasets.__version__}')
if 'qwen3_5' not in CONFIG_MAPPING:
    raise SystemExit('Transformers 5.5.0 loaded, but qwen3_5 is missing. Restart the runtime and rerun this setup cell.')
print('qwen3_5 support OK')
PY
git clone https://github.com/vasarlalikhilavinash/blindspot-env || (cd blindspot-env && git pull)
cd blindspot-env
for f in \
    data/user_summaries.json \
    data/user_splits.json \
    data/concept_catalog.json \
    data/concept_pool_per_user.json \
    data/ground_truth_adoption.json \
    data/comprehension_scores.json \
    data/reading_paths.json \
    data/novelty_flags.json; do
    test -f "$f" || { echo "Missing required real-data artifact: $f" >&2; exit 1; }
done
mkdir -p plots training/checkpoints
(nohup uvicorn server.app:app --host 0.0.0.0 --port 8000 --log-level warning > /tmp/blindspot.log 2>&1 &) || true
sleep 6
curl -s http://localhost:8000/state | head -c 300 || true
"""
)

code(
    """
import json
from pathlib import Path
import pandas as pd

DATA = Path('blindspot-env/data')
splits = json.load(open(DATA / 'user_splits.json'))
users = json.load(open(DATA / 'user_summaries.json'))
catalog = json.load(open(DATA / 'concept_catalog.json'))
paths = json.load(open(DATA / 'reading_paths.json'))
adoption = json.load(open(DATA / 'ground_truth_adoption.json'))
comp = json.load(open(DATA / 'comprehension_scores.json'))

stats = {
    'users': len(users),
    'train_users': len(splits['train']),
    'test_users': len(splits['test']),
    'concepts': len(catalog),
    'reading_paths': len(paths),
    'adoption_pairs': sum(len(v) for v in adoption.values()),
    'comprehension_pairs': sum(len(v) for v in comp.values()),
}

display(pd.DataFrame(list(stats.items()), columns=['artifact', 'value']))
print('train users:', splits['train'])
print('test users :', splits['test'])
"""
)

md("## 2. Real-data calibration")

code(
    """
%%bash
cd blindspot-env
python scripts/baseline_eval.py
"""
)

code(
    """
import json
import pandas as pd

cal = json.load(open('blindspot-env/data/baseline_calibration.json'))
rows = []
for name, vals in cal.items():
    if name.startswith('_'):
        continue
    rows.append({
        'policy': name,
        'mean_reward': round(vals['mean'], 3),
        'std': round(vals['std'], 3),
        'n': vals['n'],
    })
display(pd.DataFrame(rows))
"""
)

md(
    """
## 3. SFT warm-start

Before GRPO, build a small oracle demonstration set from the 13 training users.
This gives the model a clean action prior instead of starting from raw JSON emission.
"""
)

code(
    """
import json
import sys
from pathlib import Path

sys.path.insert(0, 'blindspot-env')

from models import BlindspotAction
from server.blindspot_environment import BlindspotEnvironment
from training.generate_sft_traces import SYSTEM_PROMPT, render_obs

env = BlindspotEnvironment()
data = env._data
splits = json.load(open('blindspot-env/data/user_splits.json'))
train_users = splits['train']
out_path = Path('blindspot-env/training/sft_traces.jsonl')


def _lookup(mapping, key, default=0.0):
    if key in mapping:
        return mapping[key]
    skey = str(key)
    if skey in mapping:
        return mapping[skey]
    return default


def score_concept(uid, cid):
    adoption = data.adoption.get(uid, {})
    comprehension = data.comprehension.get(uid, {})
    novelty = data.novelty
    a = float(_lookup(adoption, cid, 0.0))
    if a < 1e-6:
        return -0.1
    bonus = 0.5 * (1.0 if bool(_lookup(novelty, cid, False)) else 0.0)
    onboarding = float(_lookup(comprehension, cid, 0.0))
    return a + bonus + onboarding


n_traces = 0
with out_path.open('w', encoding='utf-8') as handle:
    for uid in train_users:
        for seed in range(4):
            obs = env.reset(seed=seed, user_id=uid)
            ranked = sorted(
                [c.concept_id for c in obs.candidate_concepts],
                key=lambda cid: score_concept(uid, cid),
                reverse=True,
            )
            inspect_targets = ranked[:3]
            surface_targets = ranked[:3]

            messages = [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': render_obs(obs)},
            ]

            for cid in inspect_targets:
                action = {'type': 'inspect', 'concept_id': cid}
                messages.append({'role': 'assistant', 'content': json.dumps(action)})
                obs = env.step(BlindspotAction(**action))
                if obs.done:
                    break
                messages.append({'role': 'user', 'content': render_obs(obs)})

            if not obs.done:
                for cid in surface_targets:
                    action = {'type': 'surface', 'concept_id': cid}
                    messages.append({'role': 'assistant', 'content': json.dumps(action)})
                    obs = env.step(BlindspotAction(**action))
                    if obs.done:
                        break
                    messages.append({'role': 'user', 'content': render_obs(obs)})

            if not obs.done:
                action = {'type': 'stop'}
                messages.append({'role': 'assistant', 'content': json.dumps(action)})
                obs = env.step(BlindspotAction(**action))

            handle.write(json.dumps({
                'user_id': uid,
                'seed': seed,
                'messages': messages,
                'final_reward': float(obs.reward_breakdown.total if obs.reward_breakdown else obs.reward),
            }) + '\\n')
            n_traces += 1

print(f'wrote {n_traces} oracle traces to {out_path}')
"""
)

code(
    """
%%bash
cd blindspot-env
python training/sft_train.py --backend transformers --base-model unsloth/Qwen3.5-9B --epochs 1 --batch-size 1
"""
)

md("## 4. GRPO training")

code(
    """
import importlib.metadata as importlib_metadata
import os
import sys
import types as _types

import numpy as np
import torch

loaded_numpy = np.__version__
installed_numpy = importlib_metadata.version('numpy')
if loaded_numpy != installed_numpy:
    raise RuntimeError(
        f'numpy was upgraded during setup (loaded={loaded_numpy}, installed={installed_numpy}). '
        'Restart the runtime once, then rerun from the top.'
    )

# torchcodec is baked into the Colab base image as a broken package (incompatible with torch 2.11).
# Strategy: (1) create a fake dist-info on disk so importlib.metadata.version() resolves without
# any function patching (no patching = no recursion), (2) stub sys.modules so the broken .so is
# never dlopen'd. Both blocks are fully idempotent - safe to re-run any number of times.
import importlib as _il
import importlib.metadata as _imeta
import importlib.util as _ilu
import pathlib as _pl
import tempfile as _tf
import types as _types

try:
    _imeta.version('torchcodec')
except _imeta.PackageNotFoundError:
    _fake_base = _pl.Path(_tf.mkdtemp())
    _di = _fake_base / 'torchcodec-0.0.0.dist-info'
    _di.mkdir()
    (_di / 'METADATA').write_text('Metadata-Version: 2.1\\nName: torchcodec\\nVersion: 0.0.0\\n')
    (_di / 'RECORD').write_text('')
    sys.path.insert(0, str(_fake_base))
    _il.invalidate_caches()

for _n in (
    'torchcodec',
    'torchcodec._core',
    'torchcodec._core.ops',
    'torchcodec.decoders',
):
    _m = sys.modules.get(_n)
    if _m is None:
        _m = _types.ModuleType(_n)
        sys.modules[_n] = _m
    if getattr(_m, '__spec__', None) is None:
        _m.__spec__ = _ilu.spec_from_loader(_n, loader=None)
    _m.__path__ = []  # mark as package so submodule imports don't raise TypeError

# Dummy classes datasets imports from torchcodec.decoders
sys.modules['torchcodec.decoders'].AudioDecoder = type('AudioDecoder', (), {})
sys.modules['torchcodec.decoders'].VideoDecoder = type('VideoDecoder', (), {})
sys.modules['torchcodec'].load_torchcodec_shared_libraries = lambda: None

del _n, _m, _il, _imeta, _ilu, _pl, _tf, _types

import unsloth
from unsloth import FastLanguageModel
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

BASE_MODEL = os.environ.get('BASE_MODEL', 'unsloth/Qwen3.5-9B')
FALLBACK_BASE_MODEL = os.environ.get('FALLBACK_BASE_MODEL', 'unsloth/Qwen3.5-4B')
MAX_SEQ_LENGTH = 4096 + 128

if 'qwen3_5' not in CONFIG_MAPPING:
    raise RuntimeError('This runtime still has old Transformers without qwen3_5 support. Restart runtime and rerun the setup cell.')


def load_base_model(model_name):
    return FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        fast_inference=False,
    )


try:
    model, tokenizer = load_base_model(BASE_MODEL)
    print(f'Loaded base model: {BASE_MODEL}')
except (RuntimeError, OSError) as exc:
    message = str(exc)
    missing_config = 'No config file found' in message or 'is not a local folder' in message
    unsupported_arch = 'qwen3_5' in message and ('does not support' in message or 'not recognize' in message)
    if unsupported_arch:
        raise RuntimeError('Transformers v5 is required for Qwen3.5. Restart runtime and rerun the setup cell.') from exc
    if not missing_config or BASE_MODEL == FALLBACK_BASE_MODEL:
        raise
    print(f'Base model could not be loaded: {BASE_MODEL}')
    print(f'First error line: {message.splitlines()[0]}')
    BASE_MODEL = FALLBACK_BASE_MODEL
    print(f'Falling back to: {BASE_MODEL}')
    model, tokenizer = load_base_model(BASE_MODEL)

# Attach LoRA adapters so GRPO has trainable parameters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=42,
)

if os.path.isdir('blindspot-env/training/checkpoints/sft'):
    model.load_adapter('blindspot-env/training/checkpoints/sft')
    print('attached SFT adapter')

FastLanguageModel.for_training(model)

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f'GPU: {torch.cuda.get_device_name(0)} | free {free/1e9:.1f}/{total/1e9:.1f} GB')
"""
)

code(
    """
# Build prompt dataset by sampling only train-split (user, seed) pairs from the env.
import sys, os
_bd = next(p for p in ['/content/blindspot-env', 'blindspot-env'] if os.path.isdir(p))
if _bd not in sys.path:
    sys.path.insert(0, _bd)
import json
import random
import requests
from datasets import Dataset
from training.grpo_train import SYSTEM_PROMPT, render_obs, parse_action

ENV_URL = 'http://localhost:8000'
split = json.load(open('blindspot-env/data/user_splits.json'))
train_users = split['train']
test_users = split['test']

rng = random.Random(0)
rows = []
for _ in range(256):
    uid = rng.choice(train_users)
    seed = rng.randrange(1_000_000)
    obs = requests.post(f'{ENV_URL}/reset', json={'user_id': uid, 'seed': seed}).json()
    obs = obs.get('observation', obs)
    rows.append({
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': render_obs(obs)},
        ],
        'user_id': uid,
        'seed': seed,
    })

ds = Dataset.from_list(rows)
print(f'built {len(ds)} GRPO prompts across {len(set(r["user_id"] for r in rows))} training users')
"""
)

code(
    """
import requests
import torch

REWARD_LOG = []
ROLLOUT_STEP_LIMIT = 8

# Unwrap multimodal Qwen2_5_VLProcessor -> inner text tokenizer to avoid load_image crash
from transformers import ProcessorMixin as _ProcessorMixin
_TXT_TOK = tokenizer.tokenizer if isinstance(tokenizer, _ProcessorMixin) else tokenizer


def _post_env(endpoint, payload):
    resp = requests.post(f'{ENV_URL}/{endpoint}', json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    obs = body.get('observation', body) or {}
    return body, obs


def _generate_completion(msgs):
    # apply_chat_template in transformers 5.x returns str; tokenize separately
    text = _TXT_TOK.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _TXT_TOK(text, return_tensors='pt').input_ids.to(model.device)
    with torch.inference_mode():
        out = model.generate(inputs, max_new_tokens=64, do_sample=False, temperature=0.0)
    return _TXT_TOK.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)


def run_episode(user_id, seed, first_completion=None, first_action=None, max_steps=ROLLOUT_STEP_LIMIT):
    payload = {}
    if user_id is not None:
        payload['user_id'] = user_id
    if seed is not None:
        payload['seed'] = seed

    _, obs = _post_env('reset', payload)
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': render_obs(obs)},
    ]

    action_trace = []
    completion_trace = []
    invalid_streak = 0
    for step_idx in range(max_steps):
        if step_idx == 0 and first_completion is not None:
            text = first_completion
            action = first_action or parse_action(first_completion)
        else:
            text = _generate_completion(msgs)
            action = parse_action(text)
        if not action or 'type' not in action:
            if step_idx == 0:
                return -0.05, {}, [], [text or '']
            msgs.append({'role': 'assistant', 'content': text or ''})
            msgs.append({'role': 'user', 'content': 'Reply with EXACTLY one JSON command.'})
            invalid_streak += 1
            if invalid_streak >= 2:
                break
            continue

        result, obs = _post_env('step', {'action': action})
        invalid_streak = 0
        action_trace.append(action)
        completion_trace.append(text or '')
        msgs.append({'role': 'assistant', 'content': text or ''})
        if result.get('done') or obs.get('done'):
            br = obs.get('reward_breakdown') or {}
            total = float(br.get('total', result.get('reward', 0.0) or 0.0))
            return total, br, action_trace, completion_trace
        msgs.append({'role': 'user', 'content': render_obs(obs)})

    result, obs = _post_env('step', {'action': {'type': 'stop'}})
    br = obs.get('reward_breakdown') or {}
    total = float(br.get('total', result.get('reward', 0.0) or 0.0))
    action_trace.append({'type': 'stop'})
    completion_trace.append('{"type": "stop"}  # auto-stop after rollout limit')
    return total, br, action_trace, completion_trace


def reward_fn(prompts, completions, user_id=None, seed=None, **kwargs):
    rewards = []
    uids = user_id if isinstance(user_id, list) else [user_id] * len(completions)
    seeds = seed if isinstance(seed, list) else [seed] * len(completions)

    for prompt_msgs, completion, uid, sd in zip(prompts, completions, uids, seeds):
        text = completion if isinstance(completion, str) else completion[-1].get('content', '')
        action = parse_action(text)
        if not action or 'type' not in action:
            rewards.append(-0.05)
            REWARD_LOG.append((-0.05, {}))
            continue

        total, br, _, _ = run_episode(uid, sd, first_completion=text, first_action=action)
        rewards.append(total)
        REWARD_LOG.append((total, br))

    return rewards
"""
)

code(
    """
from trl import GRPOConfig, GRPOTrainer
from transformers import ProcessorMixin

# CRITICAL: Unsloth returns a multimodal Qwen2_5_VLProcessor for Qwen3.5, not a plain tokenizer.
# When TRL calls processing_class(text=prompts_text, return_tensors='pt'), the processor tries
# to load images from the text and crashes with UnidentifiedImageError. Pass the inner text
# tokenizer instead so the entire vision codepath is bypassed.
text_tokenizer = tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer
print(f'processing_class type: {type(text_tokenizer).__name__}  (was {type(tokenizer).__name__})')

cfg = GRPOConfig(
    output_dir='blindspot-env/training/checkpoints/grpo',
    learning_rate=5e-6,
    max_steps=150,
    num_generations=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=2048,
    max_completion_length=64,
    logging_steps=5,
    save_steps=100,
    bf16=True,
    report_to='none',
)

trainer = GRPOTrainer(
    model=model,
    processing_class=text_tokenizer,
    reward_funcs=[reward_fn],
    args=cfg,
    train_dataset=ds,
)

import os; os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import torch as _torch; _torch.cuda.empty_cache()
trainer.train()
trainer.save_model('blindspot-env/training/checkpoints/grpo')
print('✓ GRPO training complete')
"""
)

md("## 5. Held-out evaluation")

code(
    """
import pandas as pd
import torch

FastLanguageModel.for_inference(model)


def trained_episode(user_id, seed):
    _, obs = _post_env('reset', {'user_id': user_id, 'seed': seed})
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': render_obs(obs)},
    ]
    first_completion = _generate_completion(msgs)
    first_action = parse_action(first_completion)
    br_total, br, actions, completions = run_episode(
        user_id,
        seed,
        first_completion=first_completion,
        first_action=first_action,
    )
    if 'total' not in br:
        br['total'] = br_total
    return br, actions, completions


results = []
all_users = train_users + test_users
for uid in all_users:
    split_name = 'test' if uid in test_users else 'train'
    for seed in range(5):
        br, actions, completions = trained_episode(uid, seed)
        results.append({
            'split': split_name,
            'user': uid,
            'seed': seed,
            'action_trace': ' -> '.join(a.get('type', '?') for a in actions),
            'first_action_type': actions[0].get('type') if actions else None,
            'num_steps': len(actions),
            'completion': '\n\n'.join(completions),
            **br,
        })

df_eval = pd.DataFrame(results)
summary = df_eval.groupby('split')['total'].agg(['mean', 'std', 'count'])
summary.loc['all'] = {
    'mean': df_eval['total'].mean(),
    'std': df_eval['total'].std(),
    'count': len(df_eval),
}
display(summary)
"""
)

code(
    """
import json
import matplotlib.pyplot as plt
import pandas as pd

cal = json.load(open('blindspot-env/data/baseline_calibration.json'))
rows = []
for name, vals in cal.items():
    if name.startswith('_'):
        continue
    rows.append({'policy': name, 'mean': vals['mean'], 'std': vals['std']})

rows.append({
    'policy': 'GRPO (train users)',
    'mean': float(df_eval[df_eval['split'] == 'train']['total'].mean()),
    'std': float(df_eval[df_eval['split'] == 'train']['total'].std()),
})
rows.append({
    'policy': 'GRPO (held-out test)',
    'mean': float(df_eval[df_eval['split'] == 'test']['total'].mean()),
    'std': float(df_eval[df_eval['split'] == 'test']['total'].std()),
})

score_df = pd.DataFrame(rows)
display(score_df)

fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(score_df['policy'], score_df['mean'], yerr=score_df['std'], capsize=5,
       color=['#888888', '#cc7733', '#3377cc', '#aa4488', '#22aa66', '#22aa66', '#117733'])
ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax.set_ylabel('mean reward')
ax.set_title('Blindspot calibration + trained-policy performance')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()
"""
)

code(
    """
import matplotlib.pyplot as plt

heldout = df_eval[df_eval['split'] == 'test'].groupby('user')['total'].agg(['mean', 'std']).sort_values('mean')
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(heldout)), heldout['mean'], yerr=heldout['std'], capsize=4,
       color=['#cc3333' if x < 0 else '#22aa66' for x in heldout['mean']])
ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
ax.set_xticks(range(len(heldout)))
ax.set_xticklabels([str(uid)[:6] for uid in heldout.index])
ax.set_ylabel('mean reward')
ax.set_title('Held-out users only')
plt.tight_layout()
plt.show()
"""
)

md(
    """
## 6. Hub push, cache precompute, reward ablation, and Space deploy

The final section makes the demo robust for the full judging window:

1. save the trained adapter
2. upload the Hub model card
3. precompute both before/after demo caches
4. run the reward-shaping ablation over the cached trained policy
5. deploy the CPU-only Hugging Face Space
"""
)

code(
    """
model.save_pretrained('blindspot-env/training/checkpoints/grpo')
tokenizer.save_pretrained('blindspot-env/training/checkpoints/grpo')
print('✓ saved adapter and tokenizer')
"""
)

code(
    """
import os

try:
    from google.colab import userdata
    for key in ['HF_TOKEN', 'GITHUB_TOKEN', 'OPENAI_API_KEY']:
        try:
            value = userdata.get(key)
            if value:
                os.environ[key] = value
        except Exception:
            pass
except Exception:
    pass

print({key: bool(os.environ.get(key)) for key in ['HF_TOKEN', 'GITHUB_TOKEN', 'OPENAI_API_KEY']})
"""
)

code(
    """
%%bash
cd blindspot-env
python scripts/push_to_hub.py
python scripts/precompute_demo_cache.py
python scripts/reward_ablation.py || true
python scripts/deploy_to_space.py
"""
)

code(
    """
import json
from pathlib import Path
import pandas as pd

ablation_path = Path('blindspot-env/data/reward_ablation.json')
if ablation_path.exists():
    ablation = json.load(open(ablation_path))
    rows = [
        {'variant': name, 'mean_reward': round(vals['mean'], 3), 'std': round(vals['std'], 3), 'n': vals['n']}
        for name, vals in ablation.items()
    ]
    display(pd.DataFrame(rows))
else:
    print('reward_ablation.json not found; skip display')
"""
)

code(
    """
import os
import subprocess

token = os.environ.get('GITHUB_TOKEN', '')
if token:
    os.chdir('blindspot-env')
    subprocess.run(['git', 'config', 'user.email', 'colab@blindspot.ai'], check=True)
    subprocess.run(['git', 'config', 'user.name', 'Blindspot Colab'], check=True)
    subprocess.run([
        'git', 'add',
        'data/demo_cache.json',
        'data/demo_cache_pretrain.json',
        'data/reward_ablation.json',
        'training/MODEL_CARD.md',
        'notebooks/02_training.ipynb',
        'notebooks/03_demo.ipynb',
    ], check=False)
    result = subprocess.run(['git', 'commit', '-m', 'Phase E: notebook, cache, and model-card refresh'])
    if result.returncode == 0:
        remote = f'https://{token}@github.com/vasarlalikhilavinash/blindspot-env.git'
        subprocess.run(['git', 'push', remote, 'HEAD:main'], check=True)
        print('✓ pushed updated artifacts to GitHub')
    else:
        print('(nothing new to commit)')
else:
    print('GITHUB_TOKEN missing; skip push-back step')
"""
)


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

out = REPO_ROOT / 'notebooks' / '02_training.ipynb'
out.write_text(json.dumps(nb, indent=1))
print(f'wrote {out} with {len(CELLS)} cells')