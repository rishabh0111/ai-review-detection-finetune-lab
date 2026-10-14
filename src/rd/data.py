"""Load the pinned dataset and the derived evaluation sets.

Labels: 1 = AI-written (`CG` in the dataset), 0 = real (`OR`).
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "data"
CROSS_GEN_DIR = ROOT / "data" / "cross_gen"

DATASET_ID = "Flowerly/modern-fake-reviews"
DATASET_REVISION = "bf57dfbbfa532f44ab26509e644329af00ff904a"

SPLITS = ("train", "validation", "test")
SHORT_MAX_WORDS = 20
LABEL_NAMES = {0: "real", 1: "ai"}


def load_split(split: str) -> pd.DataFrame:
    """Return columns text, label (0/1), category, rating, n_words."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    df = pd.read_parquet(RAW_DIR / f"{split}-00000-of-00001.parquet")
    unknown = set(df["label"]) - {"OR", "CG"}
    if unknown:
        raise ValueError(f"unexpected labels in {split}: {unknown}")
    out = pd.DataFrame(
        {
            "text": df["text_"].astype(str),
            "label": (df["label"] == "CG").astype(int),
            "category": df["category"],
            "rating": df["rating"],
        }
    )
    out["n_words"] = out["text"].str.split().str.len()
    return out.reset_index(drop=True)


def short_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Reviews of at most SHORT_MAX_WORDS words, the hard case named on the dataset card."""
    return df[df["n_words"] <= SHORT_MAX_WORDS].reset_index(drop=True)


def load_cross_gen() -> pd.DataFrame | None:
    """The cross-generator test set, if it has been generated. Same columns plus `generator`."""
    path = CROSS_GEN_DIR / "cross_gen.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["n_words"] = df["text"].str.split().str.len()
    return df


def eval_sets() -> dict[str, pd.DataFrame]:
    """Every test set a method is scored on, by name. Never train or tune on these."""
    test = load_split("test")
    sets = {"test": test, "test_short": short_subset(test)}
    cross = load_cross_gen()
    if cross is not None:
        sets["cross_gen"] = cross
        for gen, part in cross.groupby("generator"):
            sets[f"cross_gen/{gen}"] = part.reset_index(drop=True)
    return sets
