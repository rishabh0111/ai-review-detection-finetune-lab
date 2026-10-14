"""Metrics shared by every method.

Each method produces, per review, a hard prediction (0 real / 1 AI) and, where it can, a score: higher means
more likely AI. The "caught at 5% false alarms" metric uses a threshold fixed on the validation split's real
reviews, as a shop would set it before deployment, so it is never tuned on a test set.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rd.data import ROOT

RESULTS_DIR = ROOT / "results"
FALSE_ALARM_RATE = 0.05


def threshold_at_false_alarm(val_labels, val_scores, rate: float = FALSE_ALARM_RATE) -> float:
    """Score above which at most `rate` of validation real reviews are flagged."""
    real = np.asarray(val_scores)[np.asarray(val_labels) == 0]
    return float(np.quantile(real, 1 - rate, method="higher"))


def classification_metrics(labels, preds, scores=None, threshold: float | None = None) -> dict:
    y = np.asarray(labels).astype(int)
    p = np.asarray(preds).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "n": int(len(y)),
        "n_ai": int(y.sum()),
        "accuracy": (tp + tn) / len(y),
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
    }
    if scores is not None and threshold is not None:
        s = np.asarray(scores, dtype=float)
        flagged = s > threshold
        out["caught_at_5pct_false_alarm"] = float(flagged[y == 1].mean()) if y.sum() else None
        out["false_alarm_at_that_threshold"] = float(flagged[y == 0].mean()) if (y == 0).any() else None
    return out


def write_results(method: str, results: dict, meta: dict) -> Path:
    """Write results/<method>.json as soon as a method is evaluated."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{method}.json"
    payload = {
        "method": method,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta,
        "results": results,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
