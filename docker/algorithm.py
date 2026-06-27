"""
algorithm.py — PUMA Challenge Track 2 Submission
──────────────────────────────────────────────────
Model   : EfficientNetV2-M, 3-class nucleus classifier
Config  : PUMA_nuclei_V11_reg001.yaml (wd=0.20)
Best val/f1: 0.7221 (EXP-REG-001, epoch 59)

Grand Challenge I/O
───────────────────
  Input : /input/images/melanoma-whole-slide-image/<uuid>.mha   (1024×1024 H&E ROI)
  Output: /output/melanoma-10-class-nuclei-segmentation.json
          /output/images/melanoma-tissue-mask-segmentation/<uuid>.tif

Pipeline
────────
  1. Read the input ROI image (1024×1024 H&E).
  2. Detect nuclei via Hematoxylin-channel thresholding + connected-component
     filtering (min_area=50 px², max_area=5000 px²).
  3. Crop each detected nucleus (bbox + context margin), resize to 96×96.
  4. Classify each crop with EfficientNetV2-M (3 classes: tumor / lymphocyte / other).
  5. Map 3-class output → PUMA 10-class names and write the output JSON.
  6. Write a placeholder tissue mask (all-background). A dedicated tissue
     segmentation model is not included in this submission.

Class mapping
─────────────
  0 (tumor)      → nuclei_tumor
  1 (lymphocyte) → nuclei_lymphocyte
  2 (other)      → nuclei_histiocyte   (most frequent "other" subtype, 7.4%)
"""

import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import SimpleITK as sitk
import yaml
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, "/opt/app")
sys.path.insert(0, "/opt/app/nn_architecture")

from BaseModel import BaseClassificationModel  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
INPUT_DIR   = Path("/input")
OUTPUT_DIR  = Path("/output")
CONFIG_PATH = "/opt/app/configs/PUMA_nuclei_V11_reg001.yaml"
CKPT_PATH   = "/opt/app/checkpoints/best_model.ckpt"

# 3-class → PUMA 10-class label mapping
CLASS_MAP = {
    0: "nuclei_tumor",
    1: "nuclei_lymphocyte",
    2: "nuclei_histiocyte",   # best single proxy for the merged "other" class
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CROP_SIZE     = 96   # px — must match training config data.image_size
MARGIN        = 8    # context pixels added around each detected bbox
BATCH_SIZE    = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model() -> BaseClassificationModel:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config.setdefault("data", {}).update({"batch_size": BATCH_SIZE, "num_workers": 0})
    model = BaseClassificationModel.load_from_checkpoint(
        CKPT_PATH, config=config, strict=False
    )
    model.eval()
    model.to(DEVICE)
    return model


# ── Image I/O ──────────────────────────────────────────────────────────────────

def read_input_image() -> tuple[np.ndarray, str]:
    """
    Locate and load the input ROI image.
    Returns (rgb_uint8 array of shape HxWx3, filename stem for output naming).
    """
    patterns = [
        str(INPUT_DIR / "images" / "melanoma-whole-slide-image" / "*.mha"),
        str(INPUT_DIR / "images" / "melanoma-whole-slide-image" / "*.tif"),
        str(INPUT_DIR / "images" / "melanoma-whole-slide-image" / "*.tiff"),
        str(INPUT_DIR / "images" / "melanoma-roi" / "*.mha"),
        str(INPUT_DIR / "images" / "melanoma-roi" / "*.tif"),
    ]
    found = []
    for pattern in patterns:
        found.extend(glob(pattern))

    if not found:
        raise FileNotFoundError(
            "No input image found. Checked:\n" + "\n".join(patterns)
        )

    image_path = found[0]
    stem = Path(image_path).stem
    print(f"[algorithm] Input: {image_path}")

    img_sitk = sitk.ReadImage(image_path)
    arr = sitk.GetArrayFromImage(img_sitk)  # may be (H,W,C), (C,H,W), or (H,W)

    # Normalise to (H, W, 3) uint8
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    arr = arr.astype(np.uint8)
    print(f"[algorithm] Image shape: {arr.shape}")
    return arr, stem


# ── Nucleus detection ──────────────────────────────────────────────────────────

def _otsu_threshold(gray: np.ndarray) -> int:
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    total   = gray.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b   = 0.0
    w_b     = 0.0
    max_var = 0.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        w_f    = total - w_b
        sum_b += t * hist[t]
        mb     = sum_b / w_b
        mf     = (sum_all - sum_b) / w_f
        bvar   = w_b * w_f * (mb - mf) ** 2
        if bvar > max_var:
            max_var, threshold = bvar, t
    return threshold


def _hematoxylin_channel(rgb: np.ndarray) -> np.ndarray:
    """Project RGB onto the Hematoxylin optical-density axis (Ruifrok & Johnston)."""
    od  = -np.log(np.clip(rgb.astype(np.float32) / 255.0, 1e-6, 1.0))
    h_vec = np.array([0.650, 0.704, 0.286], dtype=np.float32)
    h   = od @ h_vec
    h   = np.clip(h, 0, None)
    h   = (h / (h.max() + 1e-6) * 255).astype(np.uint8)
    return h


def detect_nuclei(rgb: np.ndarray,
                  min_area: int = 50,
                  max_area: int = 5000) -> list[dict]:
    """
    Detect nuclei via Hematoxylin-channel Otsu thresholding + connected components.
    Returns list of dicts: {bbox: (x1,y1,x2,y2), centroid: (cx,cy)}.
    Bounding boxes include MARGIN pixels of tissue context on each side.
    """
    H, W = rgb.shape[:2]
    h_ch  = _hematoxylin_channel(rgb)
    thr   = _otsu_threshold(h_ch)
    binary = h_ch > thr
    binary = ndimage.binary_fill_holes(binary)
    binary = ndimage.binary_opening(binary, structure=np.ones((2, 2)), iterations=1)

    labeled, n = ndimage.label(binary)
    nuclei = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        area   = ys.size
        if area < min_area or area > max_area:
            continue
        y1_bb, y2_bb = int(ys.min()), int(ys.max())
        x1_bb, x2_bb = int(xs.min()), int(xs.max())
        cx = (x1_bb + x2_bb) // 2
        cy = (y1_bb + y2_bb) // 2
        x1 = max(0, x1_bb - MARGIN)
        y1 = max(0, y1_bb - MARGIN)
        x2 = min(W - 1, x2_bb + MARGIN)
        y2 = min(H - 1, y2_bb + MARGIN)
        nuclei.append({"bbox": (x1, y1, x2, y2), "centroid": (cx, cy)})

    print(f"[algorithm] Detected {len(nuclei)} nuclei (thr={thr})")
    return nuclei


# ── Classification ──────────────────────────────────────────────────────────────

class NucleiCropDataset(Dataset):
    def __init__(self, rgb: np.ndarray, nuclei: list[dict]):
        self.rgb    = rgb
        self.nuclei = nuclei
        self.mean   = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std    = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.nuclei)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.nuclei[idx]["bbox"]
        crop = self.rgb[y1:y2 + 1, x1:x2 + 1]
        pil  = Image.fromarray(crop).resize((CROP_SIZE, CROP_SIZE), Image.BILINEAR)
        t    = torch.from_numpy(np.array(pil)).float().div_(255.0).permute(2, 0, 1)
        return (t - self.mean) / self.std


def classify_nuclei(model: BaseClassificationModel,
                    rgb: np.ndarray,
                    nuclei: list[dict]) -> list[dict]:
    if not nuclei:
        return nuclei

    dataset = NucleiCropDataset(rgb, nuclei)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_cls, all_prob = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_cls.extend(probs.argmax(axis=1).tolist())
            all_prob.extend(probs.max(axis=1).tolist())

    for nuc, cls_idx, prob in zip(nuclei, all_cls, all_prob):
        nuc["class_idx"] = cls_idx
        nuc["score"]     = float(prob)

    counts = {}
    for nuc in nuclei:
        name = CLASS_MAP.get(nuc["class_idx"], "?")
        counts[name] = counts.get(name, 0) + 1
    print(f"[algorithm] Classification: {counts}")
    return nuclei


# ── Output writers ─────────────────────────────────────────────────────────────

def _bbox_to_path_points(x1, y1, x2, y2) -> list[list]:
    return [
        [x1, y1, 0.5],
        [x2, y1, 0.5],
        [x2, y2, 0.5],
        [x1, y2, 0.5],
        [x1, y1, 0.5],   # close ring
    ]


def write_nuclei_json(nuclei: list[dict]) -> None:
    polygons = []
    for nuc in nuclei:
        cx, cy   = nuc["centroid"]
        x1, y1, x2, y2 = nuc["bbox"]
        cls_name = CLASS_MAP.get(nuc["class_idx"], "nuclei_histiocyte")
        polygons.append({
            "name":        cls_name,
            "seed_point":  [cx, cy, 0.5],
            "path_points": _bbox_to_path_points(x1, y1, x2, y2),
            "sub_type":    "",
            "groups":      [],
            "probability": nuc["score"],
        })

    out_path = OUTPUT_DIR / "melanoma-10-class-nuclei-segmentation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"type": "Multiple polygons", "polygons": polygons}, f, indent=2)
    print(f"[algorithm] Wrote {len(polygons)} nuclei → {out_path}")


def write_tissue_mask(h: int, w: int, stem: str) -> None:
    """
    Write a placeholder all-background tissue mask.
    Tissue classes: 0=Background, 1=Stroma, 2=BloodVessel, 3=Tumor, 4=Epidermis, 5=Necrosis.
    A dedicated tissue segmentation model is not part of this submission.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    out_dir = OUTPUT_DIR / "images" / "melanoma-tissue-mask-segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.tif"
    sitk.WriteImage(sitk.GetImageFromArray(mask), str(out_path))
    print(f"[algorithm] Wrote tissue mask → {out_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    print("[algorithm] PUMA Track 2 — EfficientNetV2-M nucleus classifier")
    print(f"[algorithm] Device: {DEVICE}")

    rgb, stem = read_input_image()
    model     = load_model()
    print("[algorithm] Model loaded")

    nuclei = detect_nuclei(rgb)
    nuclei = classify_nuclei(model, rgb, nuclei)

    write_nuclei_json(nuclei)
    write_tissue_mask(rgb.shape[0], rgb.shape[1], stem)

    print("[algorithm] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
