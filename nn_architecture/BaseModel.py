"""
BaseModel.py
────────────
Shared PyTorch Lightning base classes for segmentation and classification tasks.

Architecture
────────────
Two base classes cover the two main task types:

    BaseSegmentationModel   — spatial output (B, C, H, W); batches are (images, masks)
    BaseClassificationModel — vector output  (B, C);       batches are (images, labels)

Each class receives a backbone nn.Module built by the registry functions below.
All training logic (loss, metrics, optimizer, WandB logging) lives here so
backbone files (Unet.py, SAM.py) stay pure nn.Module with no Lightning dependency.

Segmentation backbones
───────────────────────
    "unet"              — custom vanilla U-Net  (see Unet.py)
                          config: encoder_channels
    "sam_vit_b"         — SAM ViT-Base encoder + upsampling decoder  (see SAM.py)
    "sam_vit_l"         — SAM ViT-Large encoder + upsampling decoder
    "sam_vit_h"         — SAM ViT-Huge encoder + upsampling decoder
                          config: sam_checkpoint (path to .pth file)
                                  freeze_encoder (bool, default false)
    "smp_unet_<enc>"    — SMP UNet with any timm/torchvision encoder (pretrained ImageNet)
                          examples: smp_unet_resnet34, smp_unet_efficientnet-b4,
                                    smp_unet_resnext50_32x4d, smp_unet_mit_b2
                          config: pretrained (true = ImageNet weights, false = random)
                          requires: pip install segmentation-models-pytorch

Classification backbones
─────────────────────────
    "efficientnet_v2_s"  — EfficientNetV2-S   (torchvision)
    "efficientnet_v2_m"  — EfficientNetV2-M
    "efficientnet_v2_l"  — EfficientNetV2-L
    "vit_b_16"           — ViT-B/16
    "vit_l_16"           — ViT-L/16
    "resnext50_32x4d"    — ResNeXt-50 32×4d
    "resnext101_32x8d"   — ResNeXt-101 32×8d

Config keys
────────────
    model.backbone_name     — one of the keys above
    model.num_classes       — output classes (background counted for segmentation)
    model.pretrained        — true | false  (default false)
    model.loss              — "CE" (default) | "BCE"
    model.encoder_channels  — list; UNet only
    model.sam_checkpoint    — path to SAM .pth checkpoint; SAM + pretrained=true only
    model.freeze_encoder    — bool; SAM only — freeze encoder, train decoder only
    model.smp_encoder_name  — alternative to embedding encoder in backbone_name; e.g. "resnet34"
    metrics                 — segmentation:    ["dice", "iou"]
                              classification:  ["accuracy", "f1", "auc"]  (any subset)
    optimizer.name          — "adam" | "adamw" | "sgd"
    optimizer.lr            — learning rate
    optimizer.weight_decay  — default 0.0
    optimizer.momentum      — SGD only, default 0.9
"""

import os
import sys

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
import matplotlib.pyplot as plt
import numpy as np

# Allow running from the project root or from nn_architecture/ directly
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "metrics"))  # for metrics

from metrics import SegmentationMetrics, ClassificationMetrics


# ── Segmentation backbone registry ────────────────────────────────────────────

_SAM_VARIANTS = {"sam_vit_b": "vit_b", "sam_vit_l": "vit_l", "sam_vit_h": "vit_h"}

def _build_segmentation_backbone(
    name: str,
    num_classes: int,
    pretrained: bool,
    encoder_channels: list,
    sam_checkpoint: str | None = None,
    freeze_encoder: bool = False,
) -> nn.Module:
    """
    Build and return a segmentation backbone nn.Module.

    Supported names:
        "unet"           — custom vanilla U-Net (Unet.py)
        "sam_vit_b/l/h"  — SAM encoder + decoder (SAM.py); lazy-imports segment_anything
        "smp_unet_<enc>" — SMP UNet with pretrained ImageNet encoder; lazy-imports
                           segmentation_models_pytorch.  <enc> is any encoder name
                           supported by SMP (e.g. resnet34, efficientnet-b4, mit_b2).

    For SAM variants:
        - pretrained=True requires sam_checkpoint to be set in the config.
        - pretrained=False builds the SAM encoder with random weights (useful
          for debugging; not recommended for real training).
        - freeze_encoder=True freezes all SAM encoder parameters so only the
          lightweight decoder is trained.
        - segment_anything is imported lazily so the rest of the codebase does
          not depend on it.

    For SMP variants:
        - pretrained=True loads ImageNet weights for the encoder.
        - pretrained=False initialises the encoder randomly.
        - segmentation_models_pytorch is imported lazily; install separately with:
              pip install segmentation-models-pytorch
    """
    if name == "unet":
        return UNetModel(
            in_channels=3,
            num_classes=num_classes,
            encoder_channels=encoder_channels,
        )

    elif name in _SAM_VARIANTS:
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise ImportError(
                "The 'segment_anything' package is required for SAM backbones.\n"
                "Install it with:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            ) from exc

        variant = _SAM_VARIANTS[name]   # "vit_b" | "vit_l" | "vit_h"

        if pretrained:
            if not sam_checkpoint:
                raise ValueError(
                    f"pretrained=True for '{name}' requires model.sam_checkpoint "
                    f"to be set in the config.\n"
                    f"Download checkpoints from:\n"
                    f"  https://github.com/facebookresearch/segment-anything#model-checkpoints"
                )
            sam = sam_model_registry[variant](checkpoint=sam_checkpoint)
        else:
            sam = sam_model_registry[variant](checkpoint=None)

        return SAMSegmentation(sam, num_classes=num_classes, freeze_encoder=freeze_encoder)

    elif name.startswith("smp_unet_"):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for smp_unet_* backbones.\n"
                "Install it with:\n"
                "  pip install segmentation-models-pytorch"
            ) from exc

        encoder_name    = name[len("smp_unet_"):]          # e.g. "resnet34"
        encoder_weights = "imagenet" if pretrained else None
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown segmentation backbone '{name}'. "
            f"Choose from: unet, sam_vit_b, sam_vit_l, sam_vit_h, "
            f"smp_unet_<encoder> (e.g. smp_unet_resnet34)."
        )


# ── Classification backbone registry ──────────────────────────────────────────

def _build_classification_backbone(
    name: str,
    num_classes: int,
    pretrained: bool,
) -> nn.Module:
    """
    Build and return a classification backbone nn.Module.

    When pretrained=True, ImageNet weights are loaded and only the final
    fully-connected head is replaced to match num_classes.
    """
    import torchvision.models as cls

    weights = "DEFAULT" if pretrained else None

    if name == "efficientnet_v2_s":
        model = cls.efficientnet_v2_s(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "efficientnet_v2_m":
        model = cls.efficientnet_v2_m(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "efficientnet_v2_l":
        model = cls.efficientnet_v2_l(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "vit_b_16":
        model = cls.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif name == "vit_l_16":
        model = cls.vit_l_16(weights=weights)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

    elif name == "resnext50_32x4d":
        model = cls.resnext50_32x4d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnext101_32x8d":
        model = cls.resnext101_32x8d(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown classification backbone '{name}'. "
            f"Choose from: efficientnet_v2_s/m/l, vit_b_16, vit_l_16, "
            f"resnext50_32x4d, resnext101_32x8d."
        )

    return model


# ── Shared optimizer builder ───────────────────────────────────────────────────

def _build_optimizer(params, opt_cfg: dict):
    name = opt_cfg.get("name", "adam").lower()
    lr   = float(opt_cfg.get("lr", 1e-4))
    wd   = float(opt_cfg.get("weight_decay", 0.0))

    _map = {
        "adam":  torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd":   torch.optim.SGD,
    }
    if name not in _map:
        raise ValueError(f"Unknown optimizer '{name}'. Choose from {list(_map)}.")

    kwargs = {"lr": lr, "weight_decay": wd}
    if name == "sgd":
        kwargs["momentum"] = float(opt_cfg.get("momentum", 0.9))

    return _map[name](params, **kwargs)


# ── Shared metric logging helper ──────────────────────────────────────────────

def _log_metrics(model: "pl.LightningModule", val_metrics) -> None:
    """
    Call compute() on a metric accumulator and log every key to Lightning.

    Scalar keys  → logged directly with prog_bar=True.
    *_per_class keys → each element logged as val/<base>_classN (prog_bar=False).

    Works for both SegmentationMetrics and ClassificationMetrics since both
    return plain dicts from compute(). Called in on_validation_epoch_end().
    """
    if val_metrics is None:
        return
    for key, value in val_metrics.compute().items():
        if key.endswith("_per_class"):
            base = key[: -len("_per_class")]
            for c, v in enumerate(value):
                model.log(f"val/{base}_class{c}", float(v))
        else:
            model.log(f"val/{key}", float(value), prog_bar=True)


# ── BaseSegmentationModel ──────────────────────────────────────────────────────

class BaseSegmentationModel(pl.LightningModule):
    """
    Lightning base for semantic segmentation.
    Batches must be (images, masks) with masks as int64 (B, H, W).
    Backbone is selected via config["model"]["backbone_name"].
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)

        model_cfg        = config.get("model", {})
        num_classes      = int(model_cfg.get("num_classes", 6))
        backbone_name    = model_cfg.get("backbone_name", "unet")
        pretrained       = bool(model_cfg.get("pretrained", False))
        encoder_channels = model_cfg.get("encoder_channels", [64, 128, 256, 512, 1024])
        sam_checkpoint   = model_cfg.get("sam_checkpoint")
        freeze_encoder   = bool(model_cfg.get("freeze_encoder", False))

        self.model = _build_segmentation_backbone(
            backbone_name, num_classes, pretrained, encoder_channels,
            sam_checkpoint=sam_checkpoint,
            freeze_encoder=freeze_encoder,
        )
  
        # Loss
        loss_name = model_cfg.get("loss", "CE")
        self.criterion = nn.BCEWithLogitsLoss() if loss_name == "BCE" else nn.CrossEntropyLoss()

        # Metrics — only instantiate if at least one is requested
        seg_enabled  = [m for m in config.get("metrics", []) if m in ("dice", "iou")]
        self.val_metrics = (
            SegmentationMetrics(
                num_classes=num_classes,
                enabled=seg_enabled,
                ignore_background=True,
            )
            if seg_enabled else None
        )

        self._val_grid_logged = False

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (B, num_classes, H, W)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss   = self.criterion(logits, masks)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        logits = self(images)
        loss   = self.criterion(logits, masks)
        preds  = logits.argmax(dim=1)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        if self.val_metrics is not None:
            self.val_metrics.update(preds, masks)

        self._log_seg_grid(images, masks, preds, batch_idx)
        return loss

    def on_validation_epoch_start(self):
        self._val_grid_logged = False
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_validation_epoch_end(self):
        _log_metrics(self, self.val_metrics)

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return _build_optimizer(self.model.parameters(), self.hparams.get("optimizer", {}))

    # ── WandB prediction grid ─────────────────────────────────────────────────

    def _log_seg_grid(self, images, masks, preds, batch_idx, max_samples=4):
        """Image | ground truth | prediction — logged once per val epoch."""
        if self._val_grid_logged or batch_idx != 0:
            return
        if not isinstance(self.logger, pl.loggers.WandbLogger):
            return

        num_classes = int(self.hparams.get("model", {}).get("num_classes", 6))
        n    = min(max_samples, images.size(0))
        cmap = plt.get_cmap("tab10", num_classes)

        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[None]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i in range(n):
            img  = (images[i].cpu().float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[i, 0].imshow(img);                                          axes[i, 0].set_title("Image");        axes[i, 0].axis("off")
            axes[i, 1].imshow(masks[i].cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1); axes[i, 1].set_title("Ground truth"); axes[i, 1].axis("off")
            axes[i, 2].imshow(preds[i].cpu().numpy(), cmap=cmap, vmin=0, vmax=num_classes - 1); axes[i, 2].set_title("Prediction");   axes[i, 2].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({"val/predictions": wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)
        self._val_grid_logged = True


# ── BaseClassificationModel ────────────────────────────────────────────────────

class BaseClassificationModel(pl.LightningModule):
    """
    Lightning base for image classification.
    Batches must be (images, labels) with labels as int64 (B,).
    Backbone is selected via config["model"]["backbone_name"].

    Supported metrics (set in config["metrics"]):
        "accuracy"  — top-1 accuracy (epoch-level, exact)
        "f1"        — macro F1 + per-class F1
        "auc"       — macro AUC + per-class AUC (one-vs-rest; requires softmax probs)

    All metrics are computed via ClassificationMetrics (confusion-matrix accumulation).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)

        model_cfg     = config.get("model", {})
        num_classes   = int(model_cfg.get("num_classes", 2))
        backbone_name = model_cfg.get("backbone_name", "efficientnet_v2_s")
        pretrained    = bool(model_cfg.get("pretrained", False))

        self.model = _build_classification_backbone(backbone_name, num_classes, pretrained)

        loss_name = model_cfg.get("loss", "CE")
        self.criterion = nn.BCEWithLogitsLoss() if loss_name == "BCE" else nn.CrossEntropyLoss()

        # Metrics — only instantiate if at least one is requested
        cls_enabled  = [m for m in config.get("metrics", []) if m in ("accuracy", "f1", "auc")]
        self.val_metrics = (
            ClassificationMetrics(num_classes=num_classes, enabled=cls_enabled)
            if cls_enabled else None
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)   # (B, num_classes)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss   = self.criterion(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss   = self.criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

        if self.val_metrics is not None:
            # Provide softmax probs only when AUC is enabled (avoids storing
            # an (N, C) float array per batch when it is not needed)
            probs = torch.softmax(logits, dim=1) if self.val_metrics.needs_probs else None
            self.val_metrics.update(preds, labels, probs)

        self._log_cls_samples(images, labels, preds, batch_idx)
        return loss

    def on_validation_epoch_start(self):
        if self.val_metrics is not None:
            self.val_metrics.reset()

    def on_validation_epoch_end(self):
        _log_metrics(self, self.val_metrics)

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        return _build_optimizer(self.model.parameters(), self.hparams.get("optimizer", {}))

    # ── WandB sample grid ─────────────────────────────────────────────────────

    def _log_cls_samples(self, images, labels, preds, batch_idx, max_samples=8):
        """Log a row of images annotated with true / predicted class indices."""
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
            img = (images[i].cpu().float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            color = "green" if preds[i] == labels[i] else "red"
            axes[i].set_title(f"gt={labels[i].item()} / pred={preds[i].item()}", color=color)
            axes[i].axis("off")

        plt.tight_layout()
        self.logger.experiment.log({"val/samples": wandb.Image(fig)}, step=self.global_step)
        plt.close(fig)
