---
title: Blindspot Demo
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: true
license: mit
short_description: Surface AI/ML concepts you should be tracking but aren't.
---

# 🎯 Blindspot Demo

Live demo for the [OpenEnv Hackathon](https://github.com/vasarlalikhilavinash/blindspot-env) "Blindspot" submission.

**What this is:** an interactive comparison of 5 policies (Random / Trending / Dense Retrieval / GRPO-trained Blindspot / GPT-4 baseline) on the unknown-unknowns discovery task.

**Three modes:**
- 🎓 **Real user** — pick one of 17 ML researchers; rewards are computed against held-out adoption.
- 🚀 **Persona** — 3 hand-crafted personas; the trained-policy responses are precomputed (cached) so you see the real GRPO output instantly with zero GPU.
- ✍️ **Your paragraph** — paste a bio; the engine matches you to the closest user and runs the proxy policy.

**Code:** https://github.com/vasarlalikhilavinash/blindspot-env
