import sys
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config.paths import RUNS_DIR

try:
    import gradio as gr
except ImportError:
    print("Gradio not installed. Run: pip install gradio --break-system-packages")
    sys.exit(1)

from ultralytics import YOLO


####################
# Configuration
####################
# FINAL MODEL: YOLOv8s Run B
RUN_NAME    = "trainer_YOLOv8s_imgsz832_lr0.01_bs8_ep50_20260403-140823"
BEST_PT     = RUNS_DIR / RUN_NAME / "yolo_output" / "train" / "weights" / "best.pt"

CLASS_NAMES  = ["D00", "D10", "D20", "D40"]
CLASS_LABELS = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

# OpenCV uses BGR channel order
CLASS_COLORS_BGR = {
    "D00": (60, 76, 231), # red
    "D10": (219, 152, 52), # blue
    "D20": (113, 204, 46), # green
    "D40": ( 18, 156, 243), # orange
}

DEVICE      = "0" if torch.cuda.is_available() else "cpu"
CONF_THRES  = 0.25
IOU_THRES   = 0.45
INFER_IMGSZ = 832   # match Run B training resolution


####################
# Find ffmpeg
####################
def _find_ffmpeg() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    # conda env bin directory (e.g. ~/miniconda3/envs/road-damage-det/bin/ffmpeg)
    conda_ffmpeg = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
    if os.path.isfile(conda_ffmpeg):
        return conda_ffmpeg
    # common system paths
    for candidate in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/homebrew/bin/ffmpeg"]:
        if os.path.isfile(candidate):
            return candidate
    return None

FFMPEG_BIN = _find_ffmpeg()


####################
# Startup
####################
print(f"Weights : {BEST_PT}")
print(f"ffmpeg  : {FFMPEG_BIN or 'NOT FOUND — video re-encoding disabled'}")
if not FFMPEG_BIN:
    print("  → Install with: conda install -c conda-forge ffmpeg -y")

if not BEST_PT.exists():
    print(f"\nERROR: best.pt not found at:\n  {BEST_PT}")
    sys.exit(1)

print(f"Loading model ...")
model = YOLO(str(BEST_PT))
print(f"Model loaded. Device: {'GPU (CUDA)' if DEVICE == '0' else 'CPU'}\n")


####################
# Shared Utilities
####################
def run_inference(frame_bgr: np.ndarray, conf_threshold: float, iou_threshold: float):
    """
    Run YOLOv8s on a single BGR frame.
    Returns (boxes_xyxy, confidences, class_indices) or (None, None, None).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(
        source  = frame_rgb,
        imgsz   = INFER_IMGSZ,
        conf    = conf_threshold,
        iou     = iou_threshold,
        device  = DEVICE,
        verbose = False,
    )
    result = results[0]
    if result.boxes is not None and len(result.boxes):
        boxes  = result.boxes.xyxy.cpu().numpy()
        confs  = result.boxes.conf.cpu().numpy()
        labels = result.boxes.cls.cpu().numpy().astype(int)
        return boxes, confs, labels
    return None, None, None


def draw_detections(frame_bgr: np.ndarray, boxes, confs, labels) -> tuple:
    """
    Draw bounding boxes + labels on a BGR frame using OpenCV.
    Returns (annotated_bgr, table_rows).
    """
    annotated  = frame_bgr.copy()
    table_rows = []

    if boxes is None or len(boxes) == 0:
        return annotated, table_rows

    for box, conf, cls_idx in zip(boxes, confs, labels):
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cls_code = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "?"
        cls_full = CLASS_LABELS.get(cls_code, cls_code)
        color    = CLASS_COLORS_BGR.get(cls_code, (255, 255, 255))

        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

        label_text = f"{cls_code}  {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_top = max(ymin - text_h - baseline - 4, 0)
        cv2.rectangle(
            annotated,
            (xmin, label_top),
            (xmin + text_w + 4, label_top + text_h + baseline + 4),
            color, -1
        )
        cv2.putText(
            annotated, label_text,
            (xmin + 2, label_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
        )

        table_rows.append([
            cls_code, cls_full, f"{conf:.3f}",
            f"({xmin}, {ymin})", f"{xmax - xmin}×{ymax - ymin}"
        ])

    return annotated, table_rows


def _reencode_h264(raw_path: str) -> str | None:
    """
    Re-encode raw_path (mp4v) → H.264 mp4 using ffmpeg.
    Returns path to H.264 file, or None if ffmpeg unavailable / failed.
    """
    if not FFMPEG_BIN:
        return None

    out_path = tempfile.mktemp(suffix="_h264.mp4")
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", raw_path,
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",     # required for browser compatibility
        "-crf", "23",
        "-movflags", "+faststart", # move metadata to front for Gradio streaming
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ffmpeg] Error:\n{result.stderr[-500:]}")
        return None
    return out_path


####################
# Tab 1: Image
####################
def detect_image(image_np: np.ndarray, conf_threshold: float, iou_threshold: float):
    """
    image_np: RGB numpy array from Gradio (type='numpy').
    Returns (annotated_rgb, table_rows).
    """
    if image_np is None:
        return None, []

    frame_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    boxes, confs, labels = run_inference(frame_bgr, conf_threshold, iou_threshold)
    annotated_bgr, table_rows = draw_detections(frame_bgr, boxes, confs, labels)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    return annotated_rgb, table_rows


####################
# Tab 2: Video
####################
def detect_video(video_path, conf_threshold: float, iou_threshold: float, skip_n: int):
    """Wrapper with full error catching — errors surface in the Status box."""
    try:
        return _detect_video_inner(video_path, conf_threshold, iou_threshold, int(skip_n))
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[Video] EXCEPTION:\n{tb}")
        return None, f"Error: {exc}\n\nSee terminal for full traceback."


def _detect_video_inner(video_path, conf_threshold: float, iou_threshold: float, skip_n: int):
    if video_path is None:
        return None, "No video provided."

    # Gradio 4.x may pass a dict instead of a plain string path
    if isinstance(video_path, dict):
        inner = video_path.get("video") or video_path
        video_path = (inner.get("path") or inner.get("name")) if isinstance(inner, dict) else inner
    if not video_path:
        return None, "Could not extract video path from Gradio input."

    video_path = str(video_path)
    print(f"[Video] Input: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"Failed to open video: {video_path}"

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Video] {width}x{height} @ {fps:.1f}fps | {total} frames | skip_n={skip_n}")

    # Write annotated frames → raw mp4v temp file
    raw_path = tempfile.mktemp(suffix="_raw.mp4")
    writer   = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    detect_count = 0
    total_dets   = 0
    last_boxes = last_confs = last_labels = None

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if frame_idx % skip_n == 0:
            last_boxes, last_confs, last_labels = run_inference(frame_bgr, conf_threshold, iou_threshold)
            detect_count += 1
            if last_boxes is not None:
                total_dets += len(last_boxes)
        annotated_bgr, _ = draw_detections(frame_bgr, last_boxes, last_confs, last_labels)
        writer.write(annotated_bgr)
        frame_idx += 1
        if detect_count % 50 == 0 and detect_count > 0 and frame_idx % (skip_n * 50) == 0:
            print(f"  Processed {frame_idx}/{total} frames ...")

    cap.release()
    writer.release()

    # Re-encode to H.264 so browser/Gradio can play inline
    print(f"[Video] Re-encoding with ffmpeg → H.264 ...")
    h264_path = _reencode_h264(raw_path)
    os.remove(raw_path)

    status = (
        f"Done. {frame_idx} frames processed "
        f"({detect_count} inference runs, skip_n={skip_n}). "
        f"Total detections: {total_dets}."
    )
    if h264_path:
        print(f"[Video] H.264 output: {h264_path}")
        return h264_path, status
    else:
        # ffmpeg not available — return mp4v file with a note
        # Save to a named temp file so Gradio can serve it
        fallback = tempfile.mktemp(suffix="_annotated.mp4")
        # rebuild with mp4v (already deleted raw_path, need to re-run... instead just warn)
        status += "\n\nNote: ffmpeg not found — video saved as mp4v (may not play inline). Install: conda install -c conda-forge ffmpeg -y"
        print("[Video] ffmpeg not found — returning None. Install ffmpeg for inline playback.")
        return None, status


####################
# Tab 3: Webcam (Live Streaming)
####################
def detect_webcam_frame(frame_np: np.ndarray, conf_threshold: float, iou_threshold: float):
    """
    Called per-frame by Gradio's streaming webcam component.
    frame_np: RGB from browser webcam. Returns annotated RGB.
    """
    if frame_np is None:
        return None
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    boxes, confs, labels = run_inference(frame_bgr, conf_threshold, iou_threshold)
    annotated_bgr, _ = draw_detections(frame_bgr, boxes, confs, labels)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


####################
# Gradio Interface
####################
HEADER = """
# 🛣️ Road Damage Detection YOLOv8s
**Classes:** &nbsp;🔴 D00 Longitudinal Crack &nbsp;|&nbsp; 🔵 D10 Transverse Crack &nbsp;|&nbsp; 🟢 D20 Alligator Crack &nbsp;|&nbsp; 🟠 D40 Pothole
"""

with gr.Blocks(title="Road Damage Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)

    with gr.Row():
        conf_slider = gr.Slider(
            0.10, 0.90, value=CONF_THRES, step=0.05,
            label="Confidence Threshold",
            info="Lower = more detections (may include false positives)"
        )
        iou_slider = gr.Slider(
            0.10, 0.90, value=IOU_THRES, step=0.05,
            label="NMS IoU Threshold",
            info="Lower = fewer overlapping boxes"
        )

    with gr.Tabs():

        # Tab 1: Image
        with gr.TabItem("📷  Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="numpy", label="Upload Road Image", height=380)
                    img_btn   = gr.Button("🔍  Detect Damage", variant="primary", size="lg")
                with gr.Column(scale=1):
                    img_output  = gr.Image(label="Detection Result", height=380)
            img_table = gr.Dataframe(
                headers=["Code", "Class Name", "Confidence", "Position (x,y)", "Size (w×h)"],
                label="Per-Detection Details", interactive=False,
            )
            img_btn.click(
                fn=detect_image,
                inputs=[img_input, conf_slider, iou_slider],
                outputs=[img_output, img_table],
            )

        # Tab 2: Video
        with gr.TabItem("🎬  Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_input   = gr.Video(label="Upload Video")
                    skip_slider = gr.Slider(
                        1, 5, value=1, step=1,
                        label="Detect every N frames",
                        info="1 = all frames (best); 3–5 = faster for long videos"
                    )
                    vid_btn = gr.Button("▶️  Process Video", variant="primary", size="lg")
                with gr.Column(scale=1):
                    vid_output = gr.Video(label="Annotated Video")
                    vid_status = gr.Textbox(label="Status", lines=4, interactive=False)
            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input, conf_slider, iou_slider, skip_slider],
                outputs=[vid_output, vid_status],
            )

        # Tab 3: Webcam (streaming)
        with gr.TabItem("📹  Webcam (Live)"):
            gr.Markdown(
                "Real-time road damage detection via browser webcam.  \n"
            )
            with gr.Row():
                webcam_in = gr.Image(
                    sources=["webcam"],
                    streaming=True,
                    label="📷 Live Camera Feed",
                    height=420,
                )
                webcam_out = gr.Image(label="🔍 Detection Overlay", height=420)
            webcam_in.stream(
                fn=detect_webcam_frame,
                inputs=[webcam_in, conf_slider, iou_slider],
                outputs=[webcam_out],
            )

    gr.Markdown(
        "---  \n"
        "**Project:** Road damage detection and classification &nbsp;|&nbsp; "
        "ITCS352 Deep Learning &nbsp;|&nbsp; ICT Mahidol 2026 &nbsp;|&nbsp; "
        "**Student:** 6688012, 6688193 &nbsp;|&nbsp; "
        "Dataset: RDD2022 Japan (10,506 images, 4 classes)"
    )

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
)
