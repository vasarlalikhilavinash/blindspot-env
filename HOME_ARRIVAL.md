# 🏠 HOME_ARRIVAL — what to do when you get home

Total wall-clock to a live, judge-shareable demo URL: **~2.5 hours**, mostly Colab training time (you can step away).

---

## ⚡ TL;DR — the 3 commands you actually run

1. Open **Google Colab** → upload `notebooks/02_training.ipynb` → set GPU → **Runtime → Run all**.
2. Wait ~2 hours (GRPO training + cache precompute + auto-deploy).
3. Visit `https://huggingface.co/spaces/vasarlalikhilavinash/blindspot-demo` to confirm it's live, then submit that URL.

That's it. The notebook handles push-to-Hub, cache precompute, git push, and HF Space deployment — all driven by Colab secrets.

---

## 🔐 Colab secrets you need to set BEFORE running

In Colab → click the 🔑 (key) icon in the left sidebar → **Add new secret**. Enable each one for "Notebook access".

| Secret name | Where to get it | Why |
|---|---|---|
| `HF_TOKEN` | https://huggingface.co/settings/tokens (scope: **write**) | Push adapter to Hub + deploy Space |
| `GITHUB_TOKEN` | https://github.com/settings/personal-access-tokens (fine-grained, repo: `vasarlalikhilavinash/blindspot-env`, write contents) | Commit `data/demo_cache.json` back to the repo |
| `OPENAI_API_KEY` *(optional)* | https://platform.openai.com/api-keys | GPT-4 baseline column on the demo |

If `OPENAI_API_KEY` is missing the demo still works — just no GPT-4 column.

---

## 📋 Step-by-step

### 1. Set Colab secrets (5 min)
Add the 2-3 secrets above. Verify each shows "✓ enabled for this notebook".

### 2. Upload the training notebook (1 min)
File → Upload notebook → pick `notebooks/02_training.ipynb` from `~/MetaHackathon/blindspot-env/`.

### 3. Switch to GPU (30 sec)
Runtime → Change runtime type → **A100** (Pro), or T4 (free, slower).

### 4. Run all cells (~2 hours wall-clock)
Runtime → Run all. The notebook will:
- Section 1: install deps, clone repo, boot env (~3 min).
- Section 2: pre-training analysis + baseline plots (~2 min).
- Section 3: GRPO training, 400 steps × 8 generations (~90 min on A100).
- Section 4: post-training evaluation + comparison plots (~5 min).
- **Section 5 (the new bit):**
  - 5.1 push LoRA adapter → `https://huggingface.co/vasarlalikhilavinash/blindspot-qwen35-9b-grpo` (~2 min)
  - 5.2 precompute cache for 17 users + 3 personas (~3 min)
  - 5.3 commit `data/demo_cache.json` back to GitHub (~30 sec)
  - 5.4 deploy Gradio Space → `https://huggingface.co/spaces/vasarlalikhilavinash/blindspot-demo` (~2 min)

### 5. Verify the Space is live (~3 min after section 5 finishes)
Open the Space URL printed at the end. The first build takes ~2 minutes. Try:
- 🎓 Real user tab → pick any user → should see Blindspot beating baselines, with a 💾 **cached** badge on Blindspot's panel.
- 🚀 Persona tab → pick "🏥 Healthcare AI Lead" → same — cached trained-policy response.
- ✍️ Paragraph tab → paste anything → falls back to the proxy (no GPU on the Space, so this is expected).

### 6. Submit (1 min)
Paste the **HF Space URL** + the **GitHub repo URL** into the hackathon submission form.

---

## 🩹 If something goes wrong

| Symptom | Fix |
|---|---|
| `HF_TOKEN env var not set` | Re-check secret name spelling; reload the notebook. |
| `403` on git push | Regenerate `GITHUB_TOKEN` with **Contents: Read & write** for the repo. |
| Space build fails | Check Space → Logs tab. Most common: a missing dep — add it to `spaces/requirements.txt` and rerun `python scripts/deploy_to_space.py` locally. |
| Cache step fails (OOM) | Lower `max_seq_length` in `precompute_demo_cache.py` from 4096 to 2048. |
| Training crashes mid-way | The post-training cells will skip gracefully. You can re-run section 5 manually after fixing. |

## 🛟 Worst case: ship without the trained model

If training fails entirely, the Space *still* works — it falls back to the kNN-informed proxy, which already shows Blindspot at +4.00 vs Trending +1.20 (you saw this locally). Just push the current repo state, deploy the Space, and submit. The demo will note that the trained policy isn't loaded.

```bash
# Manual deploy without training:
cd ~/MetaHackathon/blindspot-env
HF_TOKEN=hf_xxx .venv/bin/python scripts/deploy_to_space.py
```

---

## 📦 What's in this repo (the deploy story)

| File | Role |
|---|---|
| `notebooks/02_training.ipynb` | The single thing you run in Colab. Trains + deploys end-to-end. |
| `notebooks/03_demo.ipynb` | Local Gradio demo (uses Colab `share=True` for a 72h tunnel). |
| `scripts/blindspot_demo.py` | Engine — 5 policies + cache-first lookup. |
| `scripts/precompute_demo_cache.py` | Builds `data/demo_cache.json` (run in Colab). |
| `scripts/push_to_hub.py` | Pushes adapter to HF Hub. |
| `scripts/deploy_to_space.py` | Pushes Gradio Space (free CPU, lives 3+ weeks). |
| `spaces/app.py` | Gradio entrypoint on the Space. |
| `spaces/README.md` | HF Space metadata frontmatter. |
| `spaces/requirements.txt` | CPU-only deps (gradio + numpy + openai). |
| `data/*.json` | All static data (users, concepts, paths, ground truth, kNN). |
| `data/demo_cache.json` | Trained-policy responses for 17 users + 3 personas. **Created by step 5.2.** |
