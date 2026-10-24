"""Scoring helpers: probation split, row-level and event-level Precision / Recall / F1."""
import numpy as np  # numpy gives us fast array maths (used by runs() below)

# NAB convention: the first 15% of each series is a "learning period".
# We fit our detector on it, then judge the detector only on the remaining 85%.
PROBATION_FRAC = 0.15


def split_probation(df, frac=PROBATION_FRAC):
    """Return (train, test): the first `frac` of rows for fitting, the rest for scoring."""
    # How many rows is 15% of the series? int() rounds down to a whole number of rows.
    n = int(len(df) * frac)
    # iloc slices by row POSITION. Rows are already in time order, so this is a time split:
    #   df.iloc[:n]  -> rows 0..n-1  (the past: used for training)
    #   df.iloc[n:]  -> rows n..end  (the future: used for testing)
    # We never shuffle, because shuffling would let the model "see the future" while training.
    return df.iloc[:n], df.iloc[n:]


def runs(mask):
    """List of (start, end) index pairs (inclusive) for each stretch of consecutive True values."""
    # Turn whatever we were given (list, Series, array of 0/1) into a numpy array of True/False.
    mask = np.asarray(mask, dtype=bool)
    # Add a False at both ends so a stretch touching the first or last row is still detected.
    # .astype(int) turns False/True into 0/1 so we can subtract neighbours next.
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    # np.diff gives (next value - current value) for each neighbouring pair:
    #   +1 means "0 then 1"  -> a stretch STARTS here
    #   -1 means "1 then 0"  -> a stretch has just ENDED
    #    0 means no change
    change = np.diff(padded)
    # Positions of the +1s are the start indexes (they line up with the original mask indexes).
    starts = np.where(change == 1)[0]
    # Positions of the -1s are one step AFTER the last True, so subtract 1 to get the last True row.
    ends = np.where(change == -1)[0] - 1
    # Pair them up: first start with first end, and so on. .tolist() gives plain Python ints.
    return list(zip(starts.tolist(), ends.tolist()))


def _prf(tp, fp, fn):
    """Turn raw counts into Precision, Recall and F1. (Leading _ means 'internal helper'.)"""
    # Precision: of everything we flagged, how much was really anomalous?  tp / (tp + fp)
    # The "if tp + fp else 0.0" guard avoids dividing by zero when we flagged nothing.
    precision = tp / (tp + fp) if tp + fp else 0.0
    # Recall: of everything really anomalous, how much did we catch?  tp / (tp + fn)
    recall = tp / (tp + fn) if tp + fn else 0.0
    # F1: harmonic mean of the two. It is high only when BOTH precision and recall are high.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1

