#!/usr/bin/env bash
# Unattended queue for the 5090: cross-generator set, rescoring, SFT, GRPO.
# Each step has a wall-clock limit, logs to logs/overnight/<step>.log and writes its results as soon as it ends;
# a failed step is recorded in logs/overnight/status.txt and the queue moves on.
#
#   tmux new -d -s night "bash scripts/overnight.sh"

cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
mkdir -p logs/overnight
STATUS=logs/overnight/status.txt

step() {  # step <name> <timeout> <command...>
  local name=$1 limit=$2; shift 2
  echo "$(date -u +%FT%TZ) start $name" >> "$STATUS"
  timeout --kill-after=60 "$limit" "$@" > "logs/overnight/$name.log" 2>&1
  local rc=$?
  echo "$(date -u +%FT%TZ) end   $name exit=$rc" >> "$STATUS"
  sleep 10  # let the GPU memory settle before the next model loads
  return $rc
}

# 1. Cross-generator fakes: three non-Qwen families, one slice of test real reviews each.
step gen_nemotron 45m python scripts/gen_cross.py --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
  --name nemotron-3-nano-4b --slot 0 --no-thinking --enforce-eager
step gen_granite 45m python scripts/gen_cross.py --model ibm-granite/granite-4.2-8b \
  --name granite-4.2-8b --slot 1 --no-thinking --enforce-eager
step gen_gemma 45m python scripts/gen_cross.py --model RedHatAI/gemma-4-12B-it-FP8-Dynamic \
  --name gemma-4-12b --slot 2 --no-thinking --enforce-eager --text-only

# 2. Rescore the finished methods so they include the cross-generator set.
step tfidf 30m python scripts/baseline_tfidf.py
step modernbert_eval 30m python scripts/train_modernbert.py --out runs/modernbert --eval-only
step zs_qwen3_4b 30m python scripts/zero_shot.py --model Qwen/Qwen3-4B-Instruct-2507 --name zero_shot_qwen3_4b
step zs_qwen3_8_27b 60m python scripts/zero_shot.py --model RedHatAI/Qwen3.8-27B-INT4 \
  --name zero_shot_qwen3_8_27b --no-thinking --text-only --enforce-eager --gpu-memory-utilization 0.85

# 3. SFT + LoRA, then score the adapter exactly like the untuned model.
step sft_train 120m python scripts/train_sft.py --out runs/sft \
  && step sft_eval 30m python scripts/zero_shot.py --model Qwen/Qwen3-4B-Instruct-2507 --name sft_qwen3_4b \
       --lora runs/sft/adapter

# 4. GRPO + LoRA, then score it with its explanations.
step grpo_train 300m python scripts/train_grpo.py --out runs/grpo --max-steps 300 \
  && step grpo_eval 60m python scripts/eval_grpo.py --lora runs/grpo/adapter --name grpo_qwen3_4b

echo "$(date -u +%FT%TZ) queue done" >> "$STATUS"
