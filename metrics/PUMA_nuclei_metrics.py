"""
PUMA_nuclei_metrics.py
──────────────────────
Dataset-specific metric accumulator for PUMA_nuclei — 3-class nucleus classification.

Extends ClassificationMetrics to replace numeric `_per_class` arrays with named float
keys so WandB logs show `val/f1_tumor`, `val/f1_lymphocyte`, `val/f1_other` instead of
`val/f1_class0`, `val/f1_class1`, `val/f1_class2`.

Classes:
    0: tumor
    1: lymphocyte
    2: other

Usage:
    m = PUMA_nucleiMetrics(num_classes=3, enabled=["accuracy", "f1", "auc"])
    m.update(preds, labels, probs)
    results = m.compute()
    # Keys: accuracy, f1, f1_tumor, f1_lymphocyte, f1_other,
    #       auc, auc_tumor, auc_lymphocyte, auc_other
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)

from metrics import ClassificationMetrics


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES  = ["tumor", "lymphocyte", "other"]

# Classes with notably lower representation — worth monitoring separately in WandB.
# tumor is the majority class; lymphocyte and other are the minority classes.
RARE_CLASSES = {"lymphocyte", "other"}


# ── PUMA_nucleiMetrics ────────────────────────────────────────────────────────

class PUMA_nucleiMetrics(ClassificationMetrics):
    """
    Epoch-level classification metrics for PUMA_nuclei.

    Wraps ClassificationMetrics.compute() and replaces numeric *_per_class arrays with
    one named float key per class. The *_per_class array is popped so _log_metrics in
    BaseModel.py does not emit duplicate entries in WandB.

    Named keys produced (depending on enabled):
        accuracy          — top-1 accuracy (unchanged from base)
        f1                — macro F1 (unchanged)
        f1_tumor          — per-class F1 for class 0
        f1_lymphocyte     — per-class F1 for class 1
        f1_other          — per-class F1 for class 2
        auc               — macro AUC (unchanged)
        auc_tumor         — per-class AUC for class 0
        auc_lymphocyte    — per-class AUC for class 1
        auc_other         — per-class AUC for class 2
    """

    _EXPECTED_NUM_CLASSES = 3

    def __init__(self, num_classes: int = 3, enabled=("accuracy", "f1", "auc"), device="cpu"):
        if num_classes != self._EXPECTED_NUM_CLASSES:
            raise ValueError(
                f"PUMA_nucleiMetrics expects num_classes={self._EXPECTED_NUM_CLASSES}, "
                f"got {num_classes}. CLASS_NAMES has {len(CLASS_NAMES)} entries."
            )
        super().__init__(num_classes=num_classes, enabled=list(enabled), device=device)

    def compute(self) -> dict:
        result = super().compute()

        for metric in ("f1", "auc"):
            per_class_key = f"{metric}_per_class"
            if per_class_key in result:
                arr = result.pop(per_class_key)
                for idx, name in enumerate(CLASS_NAMES):
                    result[f"{metric}_{name}"] = float(arr[idx])

        return result
