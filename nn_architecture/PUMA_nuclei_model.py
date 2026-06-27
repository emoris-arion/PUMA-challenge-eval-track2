"""
PUMA_nuclei_model.py
────────────────────
Dataset-specific Lightning model for PUMA_nuclei — 3-class nucleus classification
from H&E histopathology crops (96×96 px RGB).

Extends BaseClassificationModel with:
  - Optional weighted CrossEntropyLoss (inverse-frequency proxy weights)
  - PUMA_nucleiMetrics replacing numeric per-class keys with named keys
  - WandB sample grid annotated with class names instead of indices

Classes: 0=tumor, 1=lymphocyte, 2=other

Usage:
    from PUMA_nuclei_model import PUMA_nucleiClassificationModel
    model = PUMA_nucleiClassificationModel(config)
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import pytorch_lightning as pl

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "..", "metrics"))

from BaseModel import BaseClassificationModel
from PUMA_nuclei_metrics import PUMA_nucleiMetrics


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_CLASSES  = 3
CLASS_NAMES  = ["tumor", "lymphocyte", "other"]

# Inverse-frequency proxy from annotation counts (tumor=57234, lympho=21643, other=18316).
# Normalized so most-common class = 1.0.  Replace with pixel-level counts if available.
DEFAULT_CLASS_WEIGHTS = [1.0, 2.64, 3.12]


# ── FocalLoss ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification (Lin et al., 2017).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Activated when config["model"]["loss"] = "focal".
    gamma=2.0 down-weights easy (high-confidence) examples so gradients focus
    on hard minority samples — effective for class imbalance without static weights.

    Args:
        gamma: focusing parameter. 0 = standard CE. Typical values: 1–5.
        weight: optional per-class alpha tensor (same shape as nn.CrossEntropyLoss weight).
    """

    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # per-sample CE loss (no reduction) to compute p_t
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)                       # probability of correct class
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ── Model ─────────────────────────────────────────────────────────────────────

class PUMA_nucleiClassificationModel(BaseClassificationModel):
    """
    Lightning model for PUMA_nuclei 3-class nucleus classification.

    Inherits all training/validation loop logic from BaseClassificationModel.
    Overrides:
      - criterion: weighted CE when config["model"]["class_weights"] is set
      - val_metrics: PUMA_nucleiMetrics for named per-class WandB keys
      - _log_cls_samples: adds class-name labels to the WandB grid
    """

    def __init__(self, config: dict):
        super().__init__(config)

        model_cfg = config.get("model", {})

        # Select criterion based on config["model"]["loss"]:
        #   "CE" (default) → CrossEntropyLoss, optionally weighted
        #   "focal"        → FocalLoss(gamma), class_weights ignored (set null in config)
        #
        # config["model"]["class_weights"]:
        #   null / None → unweighted (use with focal or plain CE)
        #   [1.0, 2.64, 3.12] → inverse-frequency proxy weights (weighted CE only)
        loss_name   = model_cfg.get("loss", "CE")
        weights_cfg = model_cfg.get("class_weights")

        if loss_name == "focal":
            gamma = float(model_cfg.get("focal_gamma", 2.0))
            w = torch.tensor(weights_cfg, dtype=torch.float32) if weights_cfg is not None else None
            self.criterion = FocalLoss(gamma=gamma, weight=w)
        elif weights_cfg is not None:
            w = torch.tensor(weights_cfg, dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=w)
        # else: base class already set nn.CrossEntropyLoss() with no weight

        # Override val_metrics with named per-class keys
        cls_enabled = [m for m in config.get("metrics", []) if m in ("accuracy", "f1", "auc")]
        self.val_metrics = (
            PUMA_nucleiMetrics(num_classes=NUM_CLASSES, enabled=cls_enabled)
            if cls_enabled else None
        )

    def on_fit_start(self):
        # Move criterion weights to the training device (required when weight tensor
        # was created on CPU but training runs on CUDA).
        if hasattr(self.criterion, "weight") and self.criterion.weight is not None:
            self.criterion.weight = self.criterion.weight.to(self.device)

    # ── WandB sample grid with class names ────────────────────────────────────

    def _log_cls_samples(self, images, labels, preds, batch_idx, max_samples=8):
        """Log a row of nucleus crops annotated with named class labels."""
        if batch_idx != 0:
            return
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return

        n    = min(max_samples, images.size(0))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]

        for i in range(n):
            img   = (images[i].cpu().float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            gt    = CLASS_NAMES[labels[i].item()]
            pred  = CLASS_NAMES[preds[i].item()]
            color = "green" if preds[i] == labels[i] else "red"
            axes[i].imshow(img)
            axes[i].set_title(f"gt={gt}\npred={pred}", color=color, fontsize=8)
            axes[i].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({"val/samples": wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)
