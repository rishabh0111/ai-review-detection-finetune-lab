"""Method 1: full fine-tune of ModernBERT-base as a real/AI classifier.

Trains on train, keeps the checkpoint with the best validation accuracy, scores every evaluation set and writes
results/modernbert.json. The model input carries category and rating, the same information the LLMs get.

    python scripts/train_modernbert.py --out runs/modernbert
"""

import argparse
import time

import numpy as np
import torch
import transformers
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from rd.data import DATASET_ID, DATASET_REVISION, LABEL_NAMES, eval_sets, load_split
from rd.metrics import classification_metrics, threshold_at_false_alarm, write_results
from rd.prompt import review_block

MODEL = "answerdotai/ModernBERT-base"


def to_dataset(df, tok, max_len: int) -> Dataset:
    texts = [review_block(t, c, r) for t, c, r in zip(df["text"], df["category"], df["rating"])]
    enc = tok(texts, truncation=True, max_length=max_len)
    return Dataset.from_dict({**enc, "labels": df["label"].tolist()})


def ai_scores(trainer: Trainer, ds: Dataset) -> np.ndarray:
    logits = trainer.predict(ds).predictions
    return torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1)[:, 1].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/modernbert")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=640)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    revision = HfApi().model_info(MODEL).sha
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL,
        revision=revision,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    train, val = load_split("train"), load_split("validation")
    train_ds, val_ds = to_dataset(train, tok, args.max_len), to_dataset(val, tok, args.max_len)

    def compute_metrics(p):
        return {"accuracy": float((p.predictions.argmax(-1) == p.label_ids).mean())}

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=128,
        warmup_steps=0.06,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        bf16=True,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0)
    trainer.save_model(f"{args.out}/best")
    tok.save_pretrained(f"{args.out}/best")

    val_scores = ai_scores(trainer, val_ds)
    threshold = threshold_at_false_alarm(val["label"], val_scores)
    results = {"validation": classification_metrics(val["label"], val_scores > 0.5, val_scores, threshold)}
    for name, df in eval_sets().items():
        s = ai_scores(trainer, to_dataset(df, tok, args.max_len))
        results[name] = classification_metrics(df["label"], s > 0.5, s, threshold)
    for name, r in results.items():
        print(
            f"{name:<22} n={r['n']:<5} acc {r['accuracy']:.4f}  f1_ai {r['f1_ai']:.4f}  "
            f"caught@5%FA {r['caught_at_5pct_false_alarm']:.4f}"
        )

    path = write_results(
        "modernbert",
        results,
        {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "model": MODEL,
            "model_revision": revision,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_accuracy": trainer.state.best_metric,
            "train_seconds": train_seconds,
            "threshold_5pct_false_alarm": threshold,
        },
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
