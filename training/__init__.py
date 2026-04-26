"""Training stack for the Blindspot environment.

  generate_sft_traces.py — produce SFT-quality (prompt, action) traces using
                           an oracle policy + LLM rewriting (Ollama-friendly).
  sft_train.py           — supervised fine-tune Qwen2.5-7B-Instruct on the traces
                           (MLX-LM on Apple Silicon, or transformers + peft on CUDA).
  grpo_train.py          — GRPO online fine-tune via TRL + Unsloth on a HF A100.
  eval.py                — head-to-head baselines vs SFT vs GRPO; emits plots/.
"""
