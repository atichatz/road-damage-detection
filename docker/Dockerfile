# =============================================================================
# Dockerfile — Road Damage Detection Demo (YOLOv8s Run B)
# ITCS352 Deep Learning Project | ICT Mahidol 2026 | DewPerawis
# =============================================================================
#
# What this image contains:
#   - Python 3.10 slim base
#   - PyTorch (CPU-only — lighter image, works everywhere)
#   - Ultralytics YOLOv8, Gradio, OpenCV, ffmpeg
#   - The trained model weights (best.pt, 22 MB)
#   - The Gradio demo app (src_demo/demo.py)
#
# Build (run from project root — where this Dockerfile lives):
#   docker build -t road-damage-demo .
#
# Run:
#   docker run -p 7860:7860 road-damage-demo
#   Then open: http://localhost:7860
#
# Run with GPU (requires nvidia-container-toolkit):
#   docker run --gpus all -p 7860:7860 road-damage-demo
# =============================================================================

FROM python:3.10-slim

# ── Labels ───────────────────────────────────────────────────────────────────
LABEL maintainer="Perawis Buranasing, Attic"
LABEL description="Road Damage Detection Demo — YOLOv8s Run B (mAP@0.5=0.562)"
LABEL project="ITCS352 Deep Learning | ICT Mahidol 2026"

# ── System dependencies ───────────────────────────────────────────────────────
# ffmpeg     : video re-encoding (Tab 2 video processing in the demo)
# libgl1     : required by OpenCV (cv2.imshow backend, even in headless mode)
# libglib2.0-0 : required by OpenCV
# libsm6, libxext6, libxrender1 : OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# config/paths.py auto-detects PROJECT_ROOT from its own location.
# With WORKDIR=/app and files copied to /app, PROJECT_ROOT resolves to /app.
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Split into two RUN commands to avoid PyTorch-index vs PyPI mixed-index conflicts:
#   Step 1: PyTorch CPU from the official PyTorch WHL server (large, ~600 MB)
#   Step 2: Everything else from PyPI (gradio, ultralytics, opencv, numpy)
# Using --index-url (not --extra-index-url) for torch ensures the WHL server is
# used exclusively for that layer, so PyPI packages are not confused by it.

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

RUN pip install --no-cache-dir \
    "ultralytics>=8.2,<9" \
    "opencv-python-headless>=4.8" \
    "numpy>=1.24,<2"

# Install gradio separately — it has many dependencies and must be verified
# at build time so conflicts are caught here, not at container startup.
# gradio 5.x is the current stable release (4.x is legacy and has pydantic conflicts).
RUN pip install --no-cache-dir "gradio>=5.0" \
    && python -c "import gradio; print('gradio', gradio.__version__, '— import OK')"

# ── Application source ────────────────────────────────────────────────────────
COPY config/ config/
# Copy demo app folder (demo.py + __init__.py)
COPY src_demo/ src_demo/

# ── Model weights (22 MB) ─────────────────────────────────────────────────────
# best.pt sits at the root of the build context (Docker/) and is placed at
# /app/best.pt inside the container. demo.py loads it via PROJECT_ROOT / "best.pt".
COPY best.pt best.pt

# ── Port ──────────────────────────────────────────────────────────────────────
# Gradio serves on 7860 (set in demo.launch(server_port=7860))
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
# Confirms Gradio is up within 60s of container start
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860')" \
    || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["python", "src_demo/demo.py"]
