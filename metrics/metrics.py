"""
metrics.py
──────────
Task-specific metric accumulators for training and evaluation.

Hierarchy
─────────
    BaseMetrics          (abstract)
    ├── SegmentationMetrics   — Dice + IoU via confusion matrix
    ├── ClassificationMetrics — Accuracy + F1 + AUC via scikit-learn
    └── DetectionMetrics      — mAP, AP50, AP75  [planned — stub only]

Design principles
─────────────────
1. Accumulate over batches, compute once at epoch end.
   Running averages of per-batch metrics are biased when batch sizes differ.
   All classes keep state across update() calls and produce exact epoch-level
   results in compute().

2. Lazy compute() — only the metrics listed in `enabled` are computed.
   Each metric has its own private method (_dice, _iou, _accuracy, _f1, _auc).
   compute() calls only the ones that were requested at construction time.

3. Consistent interface: update / compute / reset.
   Mirrors the torchmetrics API so any class can be swapped later without
   changing model code.

4. compute() returns a plain dict of Python floats and numpy arrays.
   Keys ending in "_per_class" hold a numpy array of length num_classes.

5. Classification metrics delegate to scikit-learn (accuracy_score, f1_score,
   roc_auc_score). Segmentation keeps efficient confusion-matrix accumulation
   (bincount) because pixel-level arrays would be prohibitively large.

Usage example
─────────────
    # Segmentation — only dice requested
    m = SegmentationMetrics(num_classes=6, enabled=["dice"], ignore_background=True)
    for preds, masks in val_loader:
        m.update(preds, masks)          # int64 (B, H, W) each
    results = m.compute(); m.reset()
    # results: {"dice", "dice_per_class"}

    # Classification — accuracy + f1 + auc
    m = ClassificationMetrics(num_classes=4, enabled=["accuracy", "f1", "auc"])
    for preds, labels, probs in val_loader:
        m.update(preds, labels, probs)  # preds/labels: int64 (B,); probs: float32 (B, C)
    results = m.compute(); m.reset()
    # results: {"accuracy", "f1", "f1_per_class", "auc", "auc_per_class"}
"""

import abc

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# ── Abstract base ──────────────────────────────────────────────────────────────

class BaseMetrics(abc.ABC):
    """
    Abstract base class for all metric accumulators.

    Subclasses must implement:
        reset()   — zero all accumulators
        update()  — ingest one batch of predictions and targets
        compute() — produce a dict of epoch-level metric values

    Calling convention (enforced by model base classes):
        on_validation_epoch_start : reset()
        validation_step           : update(...)
        on_validation_epoch_end   : results = compute()
    """

    @abc.abstractmethod
    def reset(self):
        """Zero all internal accumulators. Called at the start of each epoch."""

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Accumulate predictions and targets from one batch."""

    @abc.abstractmethod
    def compute(self) -> dict:
        """
        Compute and return epoch-level metrics for all enabled keys.

        Returns:
            dict mapping metric name → float (scalars) or np.ndarray (per-class).
            Only keys for metrics listed in self.enabled are present.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ── SegmentationMetrics ────────────────────────────────────────────────────────

class SegmentationMetrics(BaseMetrics):
    """
    Epoch-level Dice and IoU for multi-class semantic segmentation.

    Uses a running confusion matrix (bincount) for efficient pixel-level
    accumulation — storing all predictions as arrays would be prohibitive for
    large images. sklearn is not used here because the math is trivial and
    operating on the already-accumulated confusion matrix is much faster.

    Formulas (per class c):
        TP_c = confusion[c, c]
        FP_c = confusion[:, c].sum() − TP_c
        FN_c = confusion[c, :].sum() − TP_c

        Dice_c = 2·TP_c / (2·TP_c + FP_c + FN_c)
        IoU_c  =   TP_c / (  TP_c + FP_c + FN_c)

    Classes with no ground-truth pixels are zeroed and excluded from the macro
    average to avoid inflating scores on trivially absent classes.

    Args:
        num_classes        (int)       — total classes including background
        enabled            (list[str]) — subset of ["dice", "iou"] to compute
        ignore_background  (bool)      — exclude class 0 from macro avg (default True)
        device             (str)       — device for the confusion matrix tensor
    """

    _VALID_METRICS = {"dice", "iou"}

    def __init__(
        self,
        num_classes: int,
        enabled: list | tuple = ("dice", "iou"),
        ignore_background: bool = True,
        device: str = "cpu",
    ):
        unknown = set(enabled) - self._VALID_METRICS
        if unknown:
            raise ValueError(f"Unknown segmentation metrics: {unknown}. Choose from {self._VALID_METRICS}.")

        self.num_classes       = num_classes
        self.enabled           = list(enabled)
        self.ignore_background = ignore_background
        self.device            = device
        self.reset()

    # ── Accumulation ──────────────────────────────────────────────────────────

    def reset(self):
        self._cm = torch.zeros(
            self.num_classes, self.num_classes,
            dtype=torch.long,
            device=self.device,
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   — int64 (B, H, W)  argmax of model logits
            targets — int64 (B, H, W)  ground-truth class indices

        Pixels with targets outside [0, num_classes) are silently skipped
        (useful for void / ignore-index labels).
        """
        preds   = preds.detach().to(self.device).long().flatten()
        targets = targets.detach().to(self.device).long().flatten()

        valid   = (targets >= 0) & (targets < self.num_classes)
        preds, targets = preds[valid], targets[valid]

        indices = self.num_classes * targets + preds
        counts  = torch.bincount(indices, minlength=self.num_classes ** 2)
        self._cm += counts.reshape(self.num_classes, self.num_classes)

    # ── Individual metric methods ──────────────────────────────────────────────

    def _cm_parts(self):
        """Return (tp, fp, fn, support) as float numpy arrays from the confusion matrix."""
        cm      = self._cm.cpu().numpy().astype(float)
        tp      = np.diag(cm)
        fp      = cm.sum(axis=0) - tp
        fn      = cm.sum(axis=1) - tp
        support = cm.sum(axis=1)
        return tp, fp, fn, support

    def _active_classes(self, support: np.ndarray) -> list[int]:
        """Classes to include in the macro average."""
        start = 1 if self.ignore_background else 0
        return [c for c in range(start, self.num_classes) if support[c] > 0]

    def _dice(self) -> dict:
        tp, fp, fn, support = self._cm_parts()
        eps = 1e-8

        per_class = (2 * tp) / (2 * tp + fp + fn + eps)
        per_class[support == 0] = 0.0

        active = self._active_classes(support)
        macro  = float(per_class[active].mean()) if active else 0.0

        return {"dice": macro, "dice_per_class": per_class}

    def _iou(self) -> dict:
        tp, fp, fn, support = self._cm_parts()
        eps = 1e-8

        per_class = tp / (tp + fp + fn + eps)
        per_class[support == 0] = 0.0

        active = self._active_classes(support)
        macro  = float(per_class[active].mean()) if active else 0.0

        return {"iou": macro, "iou_per_class": per_class}

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Returns only the metrics listed in self.enabled:
            "dice"            — macro Dice (float)
            "dice_per_class"  — per-class Dice (np.ndarray, length num_classes)
            "iou"             — macro IoU (float)
            "iou_per_class"   — per-class IoU (np.ndarray, length num_classes)
        """
        result = {}
        if "dice" in self.enabled:
            result.update(self._dice())
        if "iou" in self.enabled:
            result.update(self._iou())
        return result

    def __repr__(self) -> str:
        return (
            f"SegmentationMetrics(num_classes={self.num_classes}, "
            f"enabled={self.enabled}, ignore_background={self.ignore_background})"
        )


# ── ClassificationMetrics ──────────────────────────────────────────────────────

class ClassificationMetrics(BaseMetrics):
    """
    Epoch-level Accuracy, macro F1, and macro AUC for image classification.

    Delegates all metric computation to scikit-learn:
        accuracy  → sklearn.metrics.accuracy_score
        f1        → sklearn.metrics.f1_score  (macro + per-class)
        auc       → sklearn.metrics.roc_auc_score  (one-vs-rest, per-class + macro)

    Predictions and targets are accumulated as numpy arrays across batches.
    Softmax probabilities are only accumulated when "auc" is in enabled
    (use the needs_probs property to check this in caller code).

    Args:
        num_classes  (int)       — number of output classes
        enabled      (list[str]) — subset of ["accuracy", "f1", "auc"] to compute
        device       (str)       — used when converting tensors to numpy (ignored
                                   after detach+cpu, kept for API consistency)
    """

    _VALID_METRICS = {"accuracy", "f1", "auc"}

    def __init__(
        self,
        num_classes: int,
        enabled: list | tuple = ("accuracy", "f1"),
        device: str = "cpu",
    ):
        unknown = set(enabled) - self._VALID_METRICS
        if unknown:
            raise ValueError(f"Unknown classification metrics: {unknown}. Choose from {self._VALID_METRICS}.")

        self.num_classes = num_classes
        self.enabled     = list(enabled)
        self.device      = device
        self.reset()

    @property
    def needs_probs(self) -> bool:
        """True when AUC is enabled. Caller should pass softmax probs to update()."""
        return "auc" in self.enabled

    # ── Accumulation ──────────────────────────────────────────────────────────

    def reset(self):
        self._preds:   list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._probs:   list[np.ndarray] = []

    def update(
        self,
        preds:   torch.Tensor,
        targets: torch.Tensor,
        probs:   torch.Tensor | None = None,
    ):
        """
        Args:
            preds   — int64 (B,)      argmax of model logits
            targets — int64 (B,)      ground-truth class indices
            probs   — float32 (B, C)  softmax probabilities (required for AUC)
        """
        self._preds.append(preds.detach().cpu().numpy().flatten().astype(np.int64))
        self._targets.append(targets.detach().cpu().numpy().flatten().astype(np.int64))
        if probs is not None:
            self._probs.append(probs.detach().cpu().numpy().astype(np.float32))

    def _arrays(self):
        """Concatenate accumulated batches into epoch-level arrays."""
        preds   = np.concatenate(self._preds)
        targets = np.concatenate(self._targets)
        probs   = np.concatenate(self._probs) if self._probs else None
        return preds, targets, probs

    # ── Individual metric methods ──────────────────────────────────────────────

    def _accuracy(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        return {"accuracy": float(accuracy_score(targets, preds))}

    def _f1(self, preds: np.ndarray, targets: np.ndarray) -> dict:
        labels    = list(range(self.num_classes))
        macro     = float(f1_score(targets, preds, average="macro",
                                   labels=labels, zero_division=0))
        per_class = f1_score(targets, preds, average=None,
                             labels=labels, zero_division=0)
        return {"f1": macro, "f1_per_class": per_class}

    def _auc(self, targets: np.ndarray, probs: np.ndarray) -> dict:
        """
        One-vs-rest AUC per class, then macro average.
        Classes absent from the epoch receive AUC = 0.0 and are excluded
        from the macro average.
        """
        labels     = list(range(self.num_classes))
        per_class  = np.zeros(self.num_classes, dtype=float)

        for c in labels:
            binary_gt = (targets == c).astype(int)
            # AUC is undefined if all samples belong to one class
            if binary_gt.sum() == 0 or binary_gt.sum() == len(binary_gt):
                continue
            try:
                per_class[c] = roc_auc_score(binary_gt, probs[:, c])
            except ValueError:
                per_class[c] = 0.0

        present = [c for c in labels if (targets == c).sum() > 0]
        macro   = float(per_class[present].mean()) if present else 0.0

        return {"auc": macro, "auc_per_class": per_class}

    # ── compute ───────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        """
        Returns only the metrics listed in self.enabled:
            "accuracy"     — top-1 accuracy (float)
            "f1"           — macro F1 (float)
            "f1_per_class" — per-class F1 (np.ndarray, length num_classes)
            "auc"          — macro AUC, one-vs-rest (float)
            "auc_per_class"— per-class AUC (np.ndarray, length num_classes)

        "auc" / "auc_per_class" are only present if probs were passed to update().
        """
        if not self._preds:
            return {}

        preds, targets, probs = self._arrays()
        result = {}

        if "accuracy" in self.enabled:
            result.update(self._accuracy(preds, targets))
        if "f1" in self.enabled:
            result.update(self._f1(preds, targets))
        if "auc" in self.enabled:
            if probs is not None:
                result.update(self._auc(targets, probs))
            # else: silently skip — probs were never provided

        return result

    def __repr__(self) -> str:
        return (
            f"ClassificationMetrics(num_classes={self.num_classes}, "
            f"enabled={self.enabled})"
        )


# ── DetectionMetrics ───────────────────────────────────────────────────────────

class DetectionMetrics(BaseMetrics):
    """
    Mean Average Precision (mAP) for object detection.  [PLANNED — NOT IMPLEMENTED]

    Expected interface once implemented:

        update(
            pred_boxes:   list[Tensor],   # per image: (N, 4) float32 xyxy
            pred_scores:  list[Tensor],   # per image: (N,)   float32 confidence
            pred_labels:  list[Tensor],   # per image: (N,)   int64 class indices
            gt_boxes:     list[Tensor],   # per image: (M, 4) float32 xyxy
            gt_labels:    list[Tensor],   # per image: (M,)   int64 class indices
        )

        compute() → {
            "mAP":          float  — mean AP over IoU thresholds 0.50:0.05:0.95
            "AP50":         float  — mean AP at IoU = 0.50
            "AP75":         float  — mean AP at IoU = 0.75
            "AP_per_class": ndarray — per-class AP at IoU = 0.50
        }

    Implementation note: use torchvision.ops.box_iou for IoU computation and
    accumulate per-class precision-recall curves across all images before
    computing AP via np.trapz at epoch end.
    """

    def reset(self):
        raise NotImplementedError("DetectionMetrics is not yet implemented.")

    def update(self, *args, **kwargs):
        raise NotImplementedError("DetectionMetrics is not yet implemented.")

    def compute(self) -> dict:
        raise NotImplementedError("DetectionMetrics is not yet implemented.")
