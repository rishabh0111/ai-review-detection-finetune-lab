# Project tracker

Single source of truth for this project. **New sessions: read this whole file first.** Before ending a
session, update it: tick boxes, move "Next steps", log decisions and runs, append to the session log.

## Status

- **Phase:** 0, setup. Repo created, dataset downloaded and committed, plan written. No code yet.
- **Blocked on the user:** setting up WSL + SSH on the RTX 5090 machine (`docs/SETUP_5090_WSL_SSH.md`).
- **Last updated:** 2026-10-14

## Constraints (set by the user; don't relitigate)

- **Claude Code only.** No Anthropic API key and no other paid API: every model in the comparison runs locally.
- **One local open model** as the large untuned reference: Qwen3.8-27B, no reasoning (see Decisions).
- **One day of work** on one RTX 5090 laptop (24 GB VRAM, 64 GB RAM), Windows + WSL2, driven over SSH from the
  work laptop. **No Kaggle** (the previous project was stopped because of it).
- **Showcase in-demand fine-tuning**, not only LoRA; results must make sense to a non-technical reader (recruiters/HR).
- **Data only from** `Flowerly/modern-fake-reviews` (pinned in `data/raw/`), plus cross-generator fakes we generate.
- Public repo on GitHub (`rishabh0111`); never commit tokens. The HF token lives on the work laptop only
  (`~/.cache/huggingface/token`).


A one-day portfolio project showcasing in-demand fine-tuning, understandable by a non-technical reader:

> **Real or AI?** AI-written fake product reviews are hard for people to spot. Train models that catch them,
> including fakes from AI models they never saw in training, and compare three fine-tuning approaches against
> untuned models, a simple baseline, and people.

Audience: recruiters/HR (semi-technical) and engineers. The demo must make sense without ML knowledge.

## Plan (one day on the RTX 5090)

### Methods compared (all on the same splits and test sets)

| # | Method | Model | Why it's in the showcase |
|---|---|---|---|
| 0 | TF-IDF + logistic regression | n/a | The bar to beat. The dataset card says bag-of-words already reaches ~99% in-distribution |
| 1 | Full fine-tune of an encoder classifier | `answerdotai/ModernBERT-base` (Apache-2.0) | Classic, cheap, strong baseline |
| 2 | Supervised fine-tuning (SFT) with LoRA | `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0) | Standard LLM fine-tuning (PEFT + TRL `SFTTrainer`) |
| 3 | GRPO, reinforcement learning with a verifiable reward | same Qwen3-4B, LoRA | The most in-demand post-training technique (DeepSeek-R1 style): the model writes its reasoning, then `real`/`ai`; reward = answer matches the label (+ format). Explanations double as the demo's "why" |
| - | Untuned references | Qwen3-4B zero-shot; **Qwen3.8-27B** zero-shot, no reasoning (local, `RedHatAI/Qwen3.8-27B-INT4`) | Shows what fine-tuning adds; Qwen3.8-27B is the highest-scoring small open model |
| - | People | ~50 test reviews guessed by a few people | The relatable number for the pitch |

### Evaluation (decided before training)

1. **In-distribution:** the dataset's own `test` split (5,660). Expected to be near-saturated (~99%); reported, but
   not the headline.
2. **Cross-generator (headline):** fakes from 2-3 open models *not* used in the dataset, generated on the 5090
   with the dataset's protocol: grounded on a held-out real review from `test` (same category + star rating,
   length-matched, word-set Jaccard < 0.6). ~500 fakes per generator, paired with the real reviews they were
   grounded on. Generators: 2-3 models that fit 24 GB, from families not used as detectors (so **not Qwen**:
   the fine-tuned model and the reference are both Qwen), chosen on the 5090 (record exact ids). Candidates:
   `ibm-granite/granite-4.2-8b`, `google/gemma-4-12B-it`, `zai-org/GLM-4.7-Flash` (4-bit AWQ),
   `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`.
3. **Short reviews:** `test` reviews of <= 20 words, the hard case named on the dataset card.
4. Metrics: accuracy, precision/recall/F1 for the AI class, and the share of AI fakes caught at a fixed 5% false
   alarm rate on real reviews (what a shop would actually tune for).

Rule: `test` and the cross-generator set are never used for training, early stopping or prompt tuning;
`validation` is used for model selection.

### Schedule

| Hours | Step |
|---|---|
| 0-1 | Environment on the 5090 (WSL, CUDA, PyTorch for RTX 50 = CUDA 12.8+ wheels), clone repo, sanity-check GPU |
| 1-1.5 | Baselines: TF-IDF (method 0), untuned Qwen3-4B zero-shot on validation + test |
| 1.5-2.5 | Generate the cross-generator test set (vLLM); ModernBERT fine-tune (method 1) runs alongside |
| 2.5-3.5 | SFT with LoRA (method 2) |
| 3.5-6 | GRPO (method 3): TRL `GRPOTrainer`, LoRA, vLLM for rollouts; a few hundred steps |
| 6-7 | Evaluate everything on all test sets; human baseline; results table |
| 7-8.5 | Demo (Gradio): "game mode" (guess real/AI, then reveal answer + model verdict + GRPO explanation) and "check mode" (paste a review); README with results |

### Deliverables

- Public GitHub repo (this one) with code, results and README.
- Fine-tuned models on the Hugging Face Hub (`rishabh0111`): ModernBERT classifier, SFT adapter, GRPO adapter.
- Gradio demo (local; optionally a Hugging Face Space).
- Results table: methods x {in-distribution, cross-generator, short reviews}.

## Phases

### 0. Setup
- [x] Public GitHub repo `rishabh0111/ai-review-detection-finetune-lab`
- [x] Dataset downloaded at a pinned revision and committed (`data/raw/`, see Data)
- [x] Split overlap check (see Data)
- [ ] 5090 machine: WSL2 + Ubuntu, NVIDIA driver, SSH from the laptop **(user, guide: `docs/SETUP_5090_WSL_SSH.md`)**
- [ ] Python env on the 5090 (uv or venv, Python 3.12, torch cu128+), `nvidia-smi` and a torch matmul test

### 1. Baselines
- [ ] TF-IDF + logistic regression
- [ ] Qwen3-4B zero-shot
- [ ] Qwen3.8-27B zero-shot, no reasoning (`RedHatAI/Qwen3.8-27B-INT4`, local on the 5090)

### 2. Cross-generator test set
- [ ] Pick 2-3 generators that fit 24 GB; generate ~500 grounded fakes each; filter (length, Jaccard < 0.6)

### 3. Fine-tunes
- [ ] ModernBERT-base full fine-tune
- [ ] Qwen3-4B SFT + LoRA
- [ ] Qwen3-4B GRPO + LoRA

### 4. Evaluation
- [ ] All methods x all test sets; human baseline; results table

### 5. Demo and ship
- [ ] Gradio demo (game mode + check mode)
- [ ] Models on the HF Hub, README with results and screenshots

## Data

- **Source:** [`Flowerly/modern-fake-reviews`](https://huggingface.co/datasets/Flowerly/modern-fake-reviews),
  **CC-BY-4.0**, revision `bf57dfbbfa532f44ab26509e644329af00ff904a` (last modified 2026-07-12). Downloaded on
  2026-10-14 and committed unchanged under `data/raw/` (card: `data/raw/README.md`). Attribution is required: keep
  the credit in README and on any model card.
- **What it is:** OR = real human Amazon-style reviews (the authentic half of the Salminen et al. Kaggle "Fake
  Reviews" dataset). CG = fakes generated by `deepseek-v4-pro`, each grounded on a paired real review (same
  category and rating, length-matched, Jaccard < 0.6), punctuation normalised to ASCII.
- **Fields:** `text_`, `label` (`OR`/`CG`), `category` (10 Amazon categories), `rating` (1-5).

| split | rows | OR | CG | median words |
|---|---|---|---|---|
| train | 32,339 | 16,167 | 16,172 | 42 |
| validation | 2,425 | 1,187 | 1,238 | 39 |
| test | 5,660 | 2,858 | 2,802 | 42 |

Ratings skew to 5 stars (~60%) in every split.

- **Checks done (2026-10-14):** no review text appears in more than one split (lowercased, whitespace-normalised
  exact match: 0 overlaps for every pair); 1 duplicate inside train, none in validation/test.
- **Known limitations (from the card):** all fakes come from one generator (hence the cross-generator test); one
  generation "angle" over-represents delivery/packaging vocabulary in the fakes; "CG" means machine-generated,
  not necessarily deceptive. The pairing between each fake and its source review is not in the release, so a fake
  and its source real review can sit in different splits (near-paraphrase leakage can't be ruled out; Jaccard < 0.6
  limits it).

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-14 | Project: detect AI-written fake reviews | User wanted a one-day portfolio piece a non-technical reader understands, with data that is current and already available. Alternatives checked on the Hub: scam/phishing (data mostly old, or labels partly machine-made), terms-of-service explainer (no dataset), clickbait (mostly non-English) |
| 2026-10-14 | Dataset `Flowerly/modern-fake-reviews`, pinned revision, committed to the repo | Fakes from a 2026-era model, CC-BY-4.0, balanced, already split; 8.9 MB so committing it makes the 5090 setup a `git clone` |
| 2026-10-14 | Compare three fine-tuning approaches (ModernBERT full FT, SFT+LoRA, GRPO+LoRA) | User asked for the most in-demand fine-tuning, not limited to LoRA. GRPO/RLVR is the current in-demand technique; the comparison shows judgement about cost vs benefit |
| 2026-10-14 | Headline metric = cross-generator detection, plus a TF-IDF baseline | Dataset card: bag-of-words reaches ~99% in-distribution, so in-distribution accuracy alone shows nothing |
| 2026-10-14 | Hardware: RTX 5090 laptop (24 GB VRAM, 64 GB RAM), Windows + WSL2, driven over SSH from the work laptop | No Kaggle: its quota (now 6 h/week) and environment issues cost most of the time in the previous project |
| 2026-10-14 | Large untuned reference = **Qwen3.8-27B, no reasoning**, local on the 5090 (`RedHatAI/Qwen3.8-27B-INT4`, 19.5 GB). Fine-tune models unchanged | User's choice after research. Constraints: Claude Code only (no API key), so no Claude/API baseline; one local open model. Evidence: Artificial Analysis Intelligence Index v4.3.2, small open models 4-40B (read 2026-10-14): Qwen3.8-27B 34 (max reasoning) / 28 / 26 / 20 (no reasoning), the top small open model. Chinese alternatives that fit 24 GB: Qwen3.5-27B 23/19, Qwen3.6-27B 21/20 (AWQ 18.7 GB), GLM-4.7-Flash 15/11 (AWQ 19.8 GB), ERNIE-4.5-21B-A3B and MiMo-V2.6-Distill-9B (not on the leaderboard). Too big for 24 GB in 4-bit: Qwen3.6-35B-A3B, Intern-S2, all 100B+ Chinese models. Non-Chinese for reference: Gemma-4-26B-A4B 17, gpt-oss-20b 9-10. No reasoning: fastest for ~8,000 reviews and comparable to the fine-tuned models, which don't reason. The index measures general ability, not AI-text detection |
| 2026-10-14 | Repo name `ai-review-detection-finetune-lab` | User's choice; neutral across the three methods |

## Open questions

- Exact cross-generator models (decide on the 5090 by what fits and generates fast).
- Push models to the HF Hub under `rishabh0111` (token exists on the work laptop at `~/.cache/huggingface/token`).

## Risks

- **Qwen3.8-27B fit on 24 GB:** 19.5 GB of 4-bit weights (includes the vision encoder) leaves ~3 GB for KV cache
  and CUDA context. Reviews are short, so cap `max_model_len` (~2048). Fallbacks in order:
  `gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090` (17.9 GB, unknown uploader: check its card), then
  `Alennnndy/Qwen3.6-27B-AWQ-4bit` (18.7 GB). Also check how its "no reasoning" mode is switched (chat-template flag).
- **RTX 50 (Blackwell, sm_120) needs CUDA 12.8+ builds** of PyTorch, vLLM and flash-attention; older wheels fail
  with "no kernel image". Check in the first hour.
- **GRPO time:** rollout generation dominates. Keep completions short (~150 tokens), use vLLM for rollouts, LoRA
  on the policy; cap steps to fit the schedule.
- **Single-generator training data:** detectors may learn DeepSeek's style; the cross-generator set measures this.
- **WSL networking / office policy** may block inbound SSH; fallback is Tailscale (see setup guide).

## Next steps

1. User: follow `docs/SETUP_5090_WSL_SSH.md` on the 5090 machine; confirm `ssh 5090` works from the work laptop.
2. Then: create the Python env on the 5090, clone this repo into the WSL filesystem (`~/`, not `/mnt/c`), run the
   GPU sanity check.
3. Write the code for step 1 (baselines) and continue down the schedule.

## Related

- Previous project: `rishabh0111/tool-calling-qlora-lab` (private), tool-calling QLoRA on Kaggle. **Stopped** on 2026-10-14 (no Kaggle runs active); its
  `TRACKER.md` has the full history. Reusable pieces: LoRA/TRL training setup, vLLM serving, eval patterns.

## Session log

- **2026-10-14** (session 1): Chose the project and dataset after checking alternatives on the Hub. Created the
  public repo, downloaded the dataset at a pinned revision, checked split overlap, wrote this tracker, the README,
  CLAUDE.md and the 5090 WSL/SSH setup guide.
- **2026-10-14** (session 1, cont.): Chose `openai/gpt-oss-20b` as the local large reference. Stopped all Kaggle work on the old project.
- **2026-10-14** (session 1, cont.): Researched 2026 open models that fit 24 GB, including all major Chinese labs, against the Artificial Analysis small-model leaderboard. User ruled out API models (Claude Code only) and chose Qwen3.8-27B, no reasoning, as the one local reference. Fine-tune models stay Qwen3-4B-Instruct-2507 and ModernBERT-base.
