# Road Damage Detection Demo - Docker

**ITCS352 Introduction to Deep Learning | ICT Mahidol | 2026**

This directory contains the standalone Gradio demo and the trained YOLOv8s Run B weights.
It supports road images, uploaded videos, and webcam frames across D00 / D10 / D20 / D40.
The dataset is not needed for inference.

## Run

Install and start Docker Desktop or Docker Engine, then from the repository root:

```bash
cd docker
docker compose up --build
```

Open **http://localhost:7860** after the model finishes loading. The first build needs
internet access to download the base image and Python dependencies. This configuration
uses CPU inference, so GPU timings reported in the project are not expected here.

Stop with Ctrl+C, then run `docker compose down`.

## Contents

```text
docker/
  Dockerfile
  docker-compose.yml
  requirements_docker.txt
  best.pt
  config/
  src_demo/
    demo.py
```

Upload your own test image or video; the sample media directory mentioned in the
original submission was not present in the supplied files.

## Troubleshooting

- **No page yet:** inspect startup output with `docker compose logs -f`.
- **Port 7860 in use:** change the host port in `docker-compose.yml`, e.g. `7861:7860`,
  and open http://localhost:7861.
- **Build fails while downloading:** check internet access and available disk space.
- **Video:** ffmpeg is installed in the image for H.264 browser playback.
- **Webcam:** the browser must have access to a camera; a local video upload is an alternative.

The Dockerfile is preserved from the coursework submission. It has not been rebuilt
as part of repository packaging; its dependency ranges are not a fully locked environment.
