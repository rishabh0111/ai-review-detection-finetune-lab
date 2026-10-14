"""Untuned LLM references: ask the model `real` or `ai` and read the answer's probability.

Score = P(ai) / (P(ai) + P(real)) from the first answer token's log-probabilities; prediction = score > 0.5.
Writes results/<name>.json and per-review scores to outputs/<name>/.

    python scripts/zero_shot.py --model Qwen/Qwen3-4B-Instruct-2507 --name zero_shot_qwen3_4b
    python scripts/zero_shot.py --model Qwen/Qwen3-4B-Instruct-2507 --name sft_qwen3_4b --lora runs/sft/adapter
    python scripts/zero_shot.py --model RedHatAI/Qwen3.8-27B-INT4 --name zero_shot_qwen3_8_27b \n        --no-thinking --text-only
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import vllm
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from rd.data import DATASET_ID, DATASET_REVISION, ROOT, eval_sets, load_split
from rd.metrics import classification_metrics, threshold_at_false_alarm, write_results
from rd.prompt import classify_messages

TOP_LOGPROBS = 20


def answer_probs(logprobs: dict) -> tuple[float, float]:
    """Sum the probability of first tokens that spell `ai` or `real` (any case, any leading space)."""
    p = {"ai": 0.0, "real": 0.0}
    for lp in logprobs.values():
        word = (lp.decoded_token or "").strip().lower()
        if word in p:
            p[word] += math.exp(lp.logprob)
    return p["ai"], p["real"]


def score(llm: LLM, df, chat_kwargs: dict, lora: LoRARequest | None = None) -> tuple[np.ndarray, int]:
    """Returns P(ai) per review and how many reviews had neither answer in the top tokens (scored 0.5)."""
    convs = [classify_messages(t, c, r) for t, c, r in zip(df["text"], df["category"], df["rating"])]
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOP_LOGPROBS)
    outs = llm.chat(convs, params, chat_template_kwargs=chat_kwargs, lora_request=lora, use_tqdm=True)
    scores, missing = [], 0
    for o in outs:
        p_ai, p_real = answer_probs(o.outputs[0].logprobs[0])
        if p_ai + p_real == 0:
            missing += 1
            scores.append(0.5)
        else:
            scores.append(p_ai / (p_ai + p_real))
    return np.asarray(scores), missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False to the chat template")
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--text-only", action="store_true", help="multimodal checkpoint: reserve no vision inputs")
    ap.add_argument("--lora", default=None, help="LoRA adapter dir (adapter_config.json + weights) to score")
    ap.add_argument("--max-lora-rank", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N reviews per set (smoke test)")
    args = ap.parse_args()

    revision = HfApi().model_info(args.model).sha
    chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}
    llm = LLM(
        model=args.model,
        revision=revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if args.text_only else {}),
        **({"enable_lora": True, "max_lora_rank": args.max_lora_rank} if args.lora else {}),
        seed=0,
    )
    lora = LoRARequest("adapter", 1, str(Path(args.lora).resolve())) if args.lora else None
    first = classify_messages("Great product.", "Books_5", 5)
    print(llm.get_tokenizer().apply_chat_template(first, tokenize=False, add_generation_prompt=True, **chat_kwargs))

    sets = {"validation": load_split("validation"), **eval_sets()}
    if args.limit:
        sets = {k: v.head(args.limit) for k, v in sets.items()}
    out_dir = ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores, missing, seconds = {}, {}, {}
    for name, df in sets.items():
        if name.startswith("cross_gen/"):  # scored as part of cross_gen
            continue
        t0 = time.time()
        all_scores[name], missing[name] = score(llm, df, chat_kwargs, lora)
        seconds[name] = round(time.time() - t0, 1)
        df.assign(score=all_scores[name]).to_parquet(out_dir / f"{name}.parquet")

    threshold = threshold_at_false_alarm(sets["validation"]["label"], all_scores["validation"])
    results = {}
    for name, df in sets.items():
        s = all_scores[name] if name in all_scores else all_scores["cross_gen"][
            (sets["cross_gen"]["generator"] == name.split("/", 1)[1]).to_numpy()
        ]
        results[name] = classification_metrics(df["label"], s > 0.5, s, threshold)
        r = results[name]
        print(
            f"{name:<22} n={r['n']:<5} acc {r['accuracy']:.4f}  f1_ai {r['f1_ai']:.4f}  "
            f"caught@5%FA {r['caught_at_5pct_false_alarm']:.4f}"
        )

    path = write_results(
        args.name,
        results,
        {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "model": args.model,
            "model_revision": revision,
            "lora": args.lora,
            "vllm": vllm.__version__,
            "chat_template_kwargs": chat_kwargs,
            "decoding": "greedy, 1 token, score = P(ai) / (P(ai) + P(real)) over top-20 first tokens",
            "no_answer_in_top_tokens": missing,
            "seconds": seconds,
            "threshold_5pct_false_alarm": threshold,
            "limit": args.limit,
        },
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
