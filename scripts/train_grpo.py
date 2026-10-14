"""Method 3: GRPO (reinforcement learning with a verifiable reward) on Qwen3-4B-Instruct with LoRA.

The model writes one or two sentences on the clues it sees, then `Answer: real` or `Answer: ai`. Reward: 1 if the
answer matches the label, plus 0.1 for a well-formed answer line. No reasoning traces are needed: the model
finds its own explanations, which the demo shows as the "why". Rollouts run in vLLM, colocated on the same GPU.

    python scripts/train_grpo.py --out runs/grpo --max-steps 300
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import peft
import torch
import transformers
import trl
import vllm
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from rd.data import DATASET_ID, DATASET_REVISION, LABEL_NAMES, load_split
from rd.prompt import explain_messages, parse_answer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def to_dataset(df) -> Dataset:
    return Dataset.from_list(
        [
            {"prompt": explain_messages(t, c, r), "answer": LABEL_NAMES[int(y)]}
            for t, c, r, y in zip(df["text"], df["category"], df["rating"], df["label"])
        ]
    )


def completion_text(completion) -> str:
    return completion[-1]["content"] if isinstance(completion, list) else completion


def correct_reward(completions, answer, **_) -> list[float]:
    return [1.0 if parse_answer(completion_text(c)) == a else 0.0 for c, a in zip(completions, answer)]


def format_reward(completions, **_) -> list[float]:
    return [0.1 if parse_answer(completion_text(c)) is not None else 0.0 for c in completions]


def config(**kwargs) -> GRPOConfig:
    """Build GRPOConfig from the kwargs it actually has (argument names moved between TRL versions)."""
    fields = {f.name for f in dataclasses.fields(GRPOConfig)}
    unknown = sorted(set(kwargs) - fields)
    if unknown:
        raise SystemExit(f"GRPOConfig has no fields {unknown}")
    return GRPOConfig(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/grpo")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=16)
    ap.add_argument("--max-completion-length", type=int, default=160)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--vllm-memory", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    revision = HfApi().model_info(MODEL).sha
    train = load_split("train").sample(frac=1.0, random_state=args.seed)

    micro_batch = 2 * args.num_generations  # 2 prompts x their generations per forward pass
    accum = args.prompts_per_step * args.num_generations // micro_batch
    cfg = config(
        output_dir=args.out,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=1.0,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_memory,
        vllm_max_model_length=1024,
        vllm_enable_sleep_mode=True,
        reward_weights=[1.0, 1.0],
        save_strategy="steps",
        save_steps=100,
        logging_steps=5,
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": torch.bfloat16, "revision": revision},
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.0,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[correct_reward, format_reward],
        args=cfg,
        train_dataset=to_dataset(train),
        peft_config=lora,
    )
    trainer.model.print_trainable_parameters()

    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0)

    adapter_dir = Path(args.out) / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    meta = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model": MODEL,
        "model_revision": revision,
        "transformers": transformers.__version__,
        "trl": trl.__version__,
        "peft": peft.__version__,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "seed": args.seed,
        **{k: v for k, v in vars(args).items() if k not in ("out", "seed")},
        "train_seconds": train_seconds,
        "log_history": trainer.state.log_history,
    }
    (Path(args.out) / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved adapter to {adapter_dir} after {train_seconds}s")


if __name__ == "__main__":
    main()
