# ── PUMA Challenge Track 2 Submission ─────────────────────────────────────────
# Model : EfficientNetV2-M — 3-class nucleus classifier
# Config: PUMA_nuclei_V11_reg001.yaml  (EXP-REG-001, wd=0.20)
# val/f1: 0.7221 (best stable final epoch, epoch 59)
# Run ID: w6379wdf (wandb)
#
# This folder is fully self-contained — build from inside submission/:
#
#   cd PUMA_task2_classification/RP-0002-agent-based-cv-calibration/submission
#   docker build -t puma-track2-submission .
#
# Test locally:
#   docker run --rm --gpus all \
#     -v /path/to/input:/input:ro \
#     -v /path/to/output:/output \
#     puma-track2-submission
#
# The algorithm reads from /input/images/melanoma-whole-slide-image/<uuid>.mha
# and writes to /output/melanoma-10-class-nuclei-segmentation.json
# ──────────────────────────────────────────────────────────────────────────────

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System libs: libGL (torchvision), libgomp (OpenCV/scipy), libglib
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1-mesa-glx libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Grand Challenge convention: non-root user
RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app

# ── Python dependencies (installed as root so packages are on the system path) ─
COPY docker/requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY nn_architecture/  /opt/app/nn_architecture/
COPY metrics/          /opt/app/metrics/

# ── Config ────────────────────────────────────────────────────────────────────
COPY configs/PUMA_nuclei_V11_reg001.yaml \
     /opt/app/configs/PUMA_nuclei_V11_reg001.yaml

# ── Model checkpoint ──────────────────────────────────────────────────────────
# Best checkpoint: EXP-REG-001 wd=0.20, epoch 59, val/f1=0.7221
COPY checkpoints/best_model.ckpt /opt/app/checkpoints/best_model.ckpt

# ── Algorithm entry point ─────────────────────────────────────────────────────
COPY docker/algorithm.py /opt/app/algorithm.py

RUN mkdir -p /input /output && chmod -R 777 /output

RUN groupadd -r user && useradd -m --no-log-init -r -g user user \
    && chown -R user:user /opt/app /input /output

USER user

ENV PYTHONPATH=/opt/app

ENTRYPOINT ["python", "/opt/app/algorithm.py"]
