"""Score the GRPO adapter: the model explains, then answers `Answer: real|ai`.

Greedy decoding. Prediction = the parsed answer; score = P(ai) / (P(ai) + P(real)) at the answer token. Saves the
explanations to outputs/<name>/ for the demo and writes results/<name>.json.

    python scripts/eval_grpo.py --lora runs/grpo/adapter --name grpo_qwen3_4b
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import vllm
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from rd.data import DATASET_ID, DATASET_REVISION, ROOT, eval_sets, load_split
from rd.metrics import classification_metrics, threshold_at_false_alarm, write_results
from rd.prompt import explain_messages, parse_answer

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
STRIP = " `*.\n"


def answer_score(token_ids, logprobs, tok) -> float | None:
    """P(ai) / (P(ai) + P(real)) at the last answer-word token, or None if there is none."""
    for i in range(len(token_ids) - 1, -1, -1):
        if tok.decode([token_ids[i]]).strip(STRIP).lower() in ("ai", "real"):
            p = {"ai": 0.0, "real": 0.0}
            for lp in logprobs[i].values():
                word = (lp.decoded_token or "").strip(STRIP).lower()
                if word in p:
                    p[word] += math.exp(lp.logprob)
            return p["ai"] / (p["ai"] + p["real"]) if p["ai"] + p["real"] else None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None, help="adapter dir; omit to evaluate the untuned model this way")
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    llm = LLM(
        model=MODEL,
        max_model_len=1024,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        **({"enable_lora": True, "max_lora_rank": 64} if args.lora else {}),
    )
    tok = llm.get_tokenizer()
    lora = LoRARequest("adapter", 1, str(Path(args.lora).resolve())) if args.lora else None
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, logprobs=20)

    sets = {"validation": load_split("validation"), **eval_sets()}
    if args.limit:
        sets = {k: v.head(args.limit) for k, v in sets.items()}
    out_dir = ROOT / "outputs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    scored, unparsed, seconds = {}, {}, {}
    for name, df in sets.items():
        if name.startswith("cross_gen/"):
            continue
        t0 = time.time()
        convs = [explain_messages(t, c, r) for t, c, r in zip(df["text"], df["category"], df["rating"])]
        outs = llm.chat(convs, params, lora_request=lora, use_tqdm=True)
        texts, preds, scores = [], [], []
        for o in outs:
            c = o.outputs[0]
            answer = parse_answer(c.text)
            s = answer_score(c.token_ids, c.logprobs, tok)
            texts.append(c.text)
            preds.append(1 if answer == "ai" else 0 if answer == "real" else int((s or 0.5) > 0.5))
            scores.append(0.5 if s is None else s)
            unparsed[name] = unparsed.get(name, 0) + (answer is None)
        seconds[name] = round(time.time() - t0, 1)
        scored[name] = df.assign(completion=texts, pred=preds, score=scores)
        scored[name].to_parquet(out_dir / f"{name}.parquet")

    val = scored["validation"]
    threshold = threshold_at_false_alarm(val["label"], val["score"])
    results = {}
    for name, df in sets.items():
        if name.startswith("cross_gen/"):
            part = scored["cross_gen"]
            part = part[part["generator"] == name.split("/", 1)[1]]
        else:
            part = scored[name]
        results[name] = classification_metrics(part["label"], part["pred"], part["score"], threshold)
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
            "model": MODEL,
            "lora": args.lora,
            "vllm": vllm.__version__,
            "decoding": f"greedy, up to {args.max_tokens} tokens; pred = parsed answer, score = P(ai) at answer token",
            "unparsed_answers": unparsed,
            "seconds": seconds,
            "threshold_5pct_false_alarm": threshold,
            "limit": args.limit,
        },
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
