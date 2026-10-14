"""Method 0: TF-IDF + logistic regression, the bar every fine-tune has to beat.

Fits on train, picks C on validation, scores every evaluation set and writes results/tfidf.json.
"""

import argparse

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from rd.data import DATASET_ID, DATASET_REVISION, eval_sets, load_split
from rd.metrics import classification_metrics, threshold_at_false_alarm, write_results


def build(c: float, seed: int):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(C=c, max_iter=2000, random_state=seed),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, val = load_split("train"), load_split("validation")

    val_acc = {}
    for c in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0):
        model = build(c, args.seed).fit(train["text"], train["label"])
        val_acc[c] = float((model.predict(val["text"]) == val["label"]).mean())
        print(f"C={c:<5} validation accuracy {val_acc[c]:.4f}")
    best_c = max(val_acc, key=val_acc.get)

    model = build(best_c, args.seed).fit(train["text"], train["label"])
    val_scores = model.predict_proba(val["text"])[:, 1]
    threshold = threshold_at_false_alarm(val["label"], val_scores)

    results = {}
    for name, df in {"validation": val, **eval_sets()}.items():
        scores = model.predict_proba(df["text"])[:, 1]
        results[name] = classification_metrics(df["label"], scores > 0.5, scores, threshold)
        r = results[name]
        print(
            f"{name:<22} n={r['n']:<5} acc {r['accuracy']:.4f}  f1_ai {r['f1_ai']:.4f}  "
            f"caught@5%FA {r['caught_at_5pct_false_alarm']:.4f}"
        )

    path = write_results(
        "tfidf",
        results,
        {
            "dataset": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "seed": args.seed,
            "sklearn": sklearn.__version__,
            "features": "word 1-2 grams, min_df=2, sublinear tf",
            "C_selected_on_validation": best_c,
            "validation_accuracy_by_C": val_acc,
            "threshold_5pct_false_alarm": threshold,
        },
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
