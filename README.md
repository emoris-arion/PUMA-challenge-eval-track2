# PUMA Challenge — Track 2 Submission

**Model:** EfficientNetV2-M — 3-class nucleus classifier
**Config:** `PUMA_nuclei_V11_reg001.yaml` (EXP-REG-001, wd=0.20)
**Best val/f1:** 0.7221 (epoch 59)

## Model Checkpoint

The trained model checkpoint is not stored in this repository.
Download `best_model.ckpt` and place it at `checkpoints/best_model.ckpt` before building:

**[Download from Google Drive](https://drive.google.com/drive/folders/11KyK7zAoLX4vSaaisw4nlKL0ZsEo2rwb?usp=sharing)**

## Build & Run

```bash
# Build
docker build -t puma-track2-submission .

# Run
docker run --rm --gpus all \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output \
  puma-track2-submission
```

The algorithm reads from `/input/images/melanoma-whole-slide-image/<uuid>.mha`
and writes to `/output/melanoma-10-class-nuclei-segmentation.json`.
