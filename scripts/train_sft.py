"""Method 2: supervised fine-tuning of Qwen3-4B-Instruct with LoRA (PEFT + TRL SFTTrainer).

The model sees the same prompt as the zero-shot run and learns to answer `real` or `ai`; loss is on the answer
only. Saves the adapter (adapter_config.json + weights only, so vLLM can load it) to <out>/adapter. Scoring is
done afterwards with scripts/zero_shot.py --lora, exactly like the untuned model.

    python scripts/train_sft.py --out runs/sft
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
from datasets import Dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from rd.data import DATASET_ID, DATASET_REVISION, LABEL_NAMES, load_split
from rd.prompt import classify_messages

MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def to_dataset(df) -> Dataset:
    return Dataset.from_list(
        [
            {
                "prompt": classify_messages(t, c, r),
                "completion": [{"role": "assistant", "content": LABEL_NAMES[int(y)]}],
            }
            for t, c, r, y in zip(df["text"], df["category"], df["rating"], df["label"])
        ]
    )


def config(**kwargs) -> SFTConfig:
    """Build SFTConfig from the kwargs it actually has (argument names moved between TRL versions)."""
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    unknown = sorted(set(kwargs) - fields)
    if unknown:
        raise SystemExit(f"SFTConfig has no fields {unknown}")
    return SFTConfig(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    revision = HfApi().model_info(MODEL).sha
    train = load_split("train").sample(frac=1.0, random_state=args.seed)
    val = load_split("validation").sample(n=500, random_state=args.seed)  # loss monitoring only

    cfg = config(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        bf16=True,
        gradient_checkpointing=True,
        max_length=768,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="no",
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        model_init_kwargs={"dtype": torch.bfloat16, "revision": revision},
    )
    lora = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=MODEL,
        args=cfg,
        train_dataset=to_dataset(train),
        eval_dataset=to_dataset(val),
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
        "torch": torch.__version__,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_rank": args.rank,
        "train_seconds": train_seconds,
        "final_eval": trainer.evaluate(),
        "log_history": trainer.state.log_history,
    }
    (Path(args.out) / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"saved adapter to {adapter_dir} after {train_seconds}s")


if __name__ == "__main__":
    main()
