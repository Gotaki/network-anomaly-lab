"""Scoring helpers: probation split, row-level and event-level Precision / Recall / F1."""
import numpy as np

PROBATION_FRAC = 0.15  # NAB convention: first 15% of each series is the "learning" period


def split_probation(df, frac=PROBATION_FRAC):
    """Return (train, test): the first `frac` of rows for fitting, the rest for scoring."""
    n = int(len(df) * frac)
    return df.iloc[:n], df.iloc[n:]


def runs(mask):
    """List of (start, end) index pairs (inclusive) for each stretch of consecutive True values."""
    mask = np.asarray(mask, dtype=bool)
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    change = np.diff(padded)
    starts = np.where(change == 1)[0]
    ends = np.where(change == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

