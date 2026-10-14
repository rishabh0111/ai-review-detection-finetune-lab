# CLAUDE.md

**Start every session by reading `TRACKER.md`** (status, plan, data, decisions, next steps). Before ending a
session, update it: tick boxes, update "Status" and "Next steps", log decisions, append to the session log.

## Environment

- Code is edited on the work laptop (Windows) and run on the RTX 5090 machine over SSH (`ssh 5090`), inside WSL2
  Ubuntu, in `~/ai-review-detection-finetune-lab` (Linux filesystem, not `/mnt/c`). Setup:
  `docs/SETUP_5090_WSL_SSH.md`.
- Long runs go inside `tmux` on the 5090 so they survive disconnects.
- RTX 50 series: PyTorch/vLLM wheels must be built for CUDA 12.8+.

## Rules for this project

- `data/raw/` is the pinned upstream dataset; never edit it. Derived data goes elsewhere.
- The `test` split and the cross-generator set are never used for training, early stopping or prompt tuning;
  use `validation` for model selection.
- Record exact model ids, dataset revisions and seeds for every run in `TRACKER.md`.
- Never print or commit tokens (HF, API keys).
