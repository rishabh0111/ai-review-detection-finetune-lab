"""Generate the cross-generator test set: fakes from a model the detectors never saw.

Follows the dataset's own protocol (DATA_PROVENANCE.md in PurplexyFlower/Fake-review): each fake is grounded on
the first 400 characters of a real review from `test`, same category and star rating, about the same length, with
one of seven "angles"; up to five attempts with rising temperature; accepted if >= 3 words, within the length
ceiling and word-set Jaccard < 0.6 with the source. Each generator gets its own slice of `test` real reviews, so
every generator's subset is balanced: its accepted fakes plus the real reviews they were grounded on.

    python scripts/gen_cross.py --model google/gemma-4-12B-it --name gemma-4-12b --slot 0

Writes data/cross_gen/parts/<name>.parquet, then rebuilds data/cross_gen/cross_gen.parquet from all parts.
"""

import argparse
import random
import re
import time

import pandas as pd
import vllm
from huggingface_hub import HfApi
from vllm import LLM, SamplingParams

from rd.data import CROSS_GEN_DIR, load_split
from rd.prompt import category_name

SYSTEM = (
    "You write realistic, concise Amazon-style customer product reviews. Output ONLY the review text — no "
    "preamble, no surrounding quotes, no title, no rating line, no markdown."
)
USER = (
    'Here is a real customer review of a product in the {category} category:\n"{source}"\n\n'
    "Write a NEW, different {rating}-star review for the SAME product as if you were another genuine customer. "
    "Emphasise {angle}. About {target} words. Natural, specific, casual. Do NOT copy phrasing from the review "
    "above. Output only the review text."
)
ANGLES = (
    "overall experience",
    "a specific feature liked or disliked",
    "value for the price",
    "comparison with expectations",
    "shipping/packaging and first impressions",
    "durability after some use",
    "intended recipient or recommendation",
)
ATTEMPTS = 5
SLOT_SIZE = 700  # real reviews per generator; ~500 accepted fakes are kept
ASCII_MAP = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "...",
     " ": " "}
)


def words(text: str) -> list[str]:
    return text.split()


def jaccard(a: str, b: str) -> float:
    sa, sb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def clean(text: str) -> str:
    """Strip boilerplate the protocol removes, and normalise punctuation to ASCII as the dataset does."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)  # reasoning models that ignore the off switch
    t = t.translate(ASCII_MAP).strip()
    t = re.sub(r"^(here(?:'s| is) (?:a|my|the) [^:\n]*review[^:\n]*:|review:)\s*", "", t, flags=re.I)
    t = re.sub(r"^\*\*[^*]+\*\*\s*", "", t)  # markdown title
    t = t.strip().strip('"').strip()
    return re.sub(r"\s+", " ", t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True, help="short generator name used in the results")
    ap.add_argument("--slot", type=int, required=True, help="which slice of test real reviews to ground on")
    ap.add_argument("--keep", type=int, default=500)
    ap.add_argument("--no-thinking", action="store_true", help="pass enable_thinking=False to the chat template")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--text-only", action="store_true", help="multimodal checkpoint: reserve no vision inputs")
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA graph capture")
    ap.add_argument("--extra", default="", help="extra LLM kwargs as key=value,key=value (e.g. quantization=awq)")
    args = ap.parse_args()

    test = load_split("test")
    reals = test[test["label"] == 0].sample(frac=1.0, random_state=1998).reset_index()  # keeps test row index
    src = reals.iloc[args.slot * SLOT_SIZE : (args.slot + 1) * SLOT_SIZE].reset_index(drop=True)
    if len(src) < SLOT_SIZE:
        raise SystemExit(f"slot {args.slot} has only {len(src)} real reviews")

    src["target"] = src["n_words"].clip(4, 120)
    src["angle"] = [ANGLES[random.Random(int(i)).randrange(len(ANGLES))] for i in src["index"]]
    convs = [
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(
                category=category_name(r.category), source=r.text[:400], rating=int(r.rating),
                angle=r.angle, target=int(r.target))},
        ]
        for r in src.itertuples()
    ]

    revision = HfApi().model_info(args.model).sha
    extra = dict(kv.split("=", 1) for kv in args.extra.split(",") if kv)
    llm = LLM(
        model=args.model,
        revision=revision,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=0,
        enforce_eager=args.enforce_eager,
        **({"limit_mm_per_prompt": {"image": 0, "video": 0}} if args.text_only else {}),
        **extra,
    )
    chat_kwargs = {"enable_thinking": False} if args.no_thinking else {}

    accepted: dict[int, tuple[str, int]] = {}
    t0 = time.time()
    for attempt in range(ATTEMPTS):
        pending = [i for i in range(len(src)) if i not in accepted]
        if not pending:
            break
        params = [
            SamplingParams(temperature=0.85 + 0.06 * attempt, top_p=0.95, max_tokens=4 * int(src.target[i]) + 100,
                           seed=1000 * attempt + i)
            for i in pending
        ]
        outs = llm.chat([convs[i] for i in pending], params, chat_template_kwargs=chat_kwargs, use_tqdm=True)
        for i, o in zip(pending, outs):
            text = clean(o.outputs[0].text)
            n = len(words(text))
            ceiling = max(130, 1.5 * int(src.target[i]))
            if n >= 3 and n <= ceiling and jaccard(text, src.text[i]) < 0.6:
                accepted[i] = (text, attempt)
        print(f"attempt {attempt}: {len(accepted)}/{len(src)} accepted")

    keep = sorted(accepted)[: args.keep]
    fakes = pd.DataFrame(
        {
            "text": [accepted[i][0] for i in keep],
            "label": 1,
            "category": src.category[keep].to_numpy(),
            "rating": src.rating[keep].to_numpy(),
            "generator": args.name,
            "pair_id": src["index"][keep].to_numpy(),
            "attempt": [accepted[i][1] for i in keep],
        }
    )
    reals_kept = pd.DataFrame(
        {
            "text": src.text[keep].to_numpy(),
            "label": 0,
            "category": src.category[keep].to_numpy(),
            "rating": src.rating[keep].to_numpy(),
            "generator": args.name,
            "pair_id": src["index"][keep].to_numpy(),
            "attempt": -1,
        }
    )
    part = pd.concat([fakes, reals_kept], ignore_index=True)
    part.attrs = {}
    part["model"] = args.model
    part["model_revision"] = revision
    part["vllm"] = vllm.__version__

    parts_dir = CROSS_GEN_DIR / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    part.to_parquet(parts_dir / f"{args.name}.parquet", index=False)
    allparts = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("*.parquet"))], ignore_index=True)
    allparts.to_parquet(CROSS_GEN_DIR / "cross_gen.parquet", index=False)

    fw = fakes["text"].str.split().str.len()
    rw = reals_kept["text"].str.split().str.len()
    print(
        f"{args.name}: kept {len(fakes)} fakes (+{len(reals_kept)} paired reals) in {time.time() - t0:.0f}s; "
        f"median words fake {fw.median():.0f} vs real {rw.median():.0f}; "
        f"accepted per attempt {fakes['attempt'].value_counts().sort_index().to_dict()}"
    )
    for t in fakes["text"].head(3):
        print("  >", t[:300])


if __name__ == "__main__":
    main()
