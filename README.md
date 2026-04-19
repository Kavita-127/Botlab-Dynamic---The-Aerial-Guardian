#  Aerial Guardian - Drone Detection & Tracking

A lightweight, CPU-only pipeline for detecting and tracking **people and vehicles** in drone footage using **YOLO11n + SAHI + ByteTrack**. Model size is ~5.4 MB, which is under the 300 MB limit.

---

##  The Problem

From drone altitude, a person might occupy just 15–20 pixels in a 1344×756 frame. Standard detectors resize the full frame to 640×640, making tiny objects invisible. On top of that, drone camera movement causes trackers to lose IDs constantly.

This pipeline solves both problems.

---

##  Summary Report

### 1. Architecture & Small Object Detection

**Model:** YOLO11n (5.4 MB), restricted to persons and vehicles only.

| Class ID | Label |
|---|---|
| 0 | Person |
| 2 | Car |
| 3 | Motorcycle |
| 5 | Bus |
| 7 | Truck |

**Small object problem:** At drone altitude, people and vehicles are too tiny for standard full-frame inference to detect reliably.

**Solution - SAHI:** The frame is sliced into overlapping **512×512 patches** (20% overlap) and YOLO runs on each patch independently. A 15-pixel person is seen at full resolution inside its local patch. Detections are merged across patches using **Non-Maximum Merging (NMM)**.

```
Full frame → Slice into 512×512 patches → YOLO11n on each → NMM merge → ByteTrack
```

Trade-off: ~8 YOLO inferences per frame instead of 1. Slower, but dramatically better recall on small objects.

---

### 2. ID Switching - Ego-Motion & Occlusions

**Ego-motion:** When the drone pans, all objects shift position — even stationary ones. This breaks IoU-based tracking and causes ID switches.

**Fix — `EgoMotionCompensator`:**
1. Extract background feature points from the previous frame (`cv2.goodFeaturesToTrack`)
2. Track them into the current frame using Lucas-Kanade optical flow
3. Compute **median translation** (dx, dy) - median ignores moving-object outliers
4. Shift all detections by (−dx, −dy) before passing to ByteTrack

This cancels out the drone's movement so the tracker sees a stable world.

**Occlusions:** ByteTrack's `lost_track_buffer` (60 frames = ~2 seconds) keeps a track alive through brief disappearances. Low-confidence detections (0.1–0.25) are retained as secondary candidates for re-association, so an object reappears with its original ID.

---

### 3. Edge Hardware - NVIDIA Jetson

| Step | Action |
|---|---|
| TensorRT export | `yolo export model=yolo11n.pt format=engine device=0` |
| INT8 quantization | Add `int8=True` to above command - doubles throughput |
| Smaller patches | Reduce slice size to 320×320 in `pipeline.py` |
| Skip SAHI | At lower altitudes where objects are larger, bypass SAHI entirely for 30+ FPS |

**Estimated FPS on Jetson:**

| Hardware | Mode | FPS |
|---|---|---|
| Jetson Nano | TensorRT FP16, no SAHI | ~15–20 |
| Jetson Orin NX | TensorRT INT8, SAHI 320 | ~10–15 |
| Jetson Orin NX | TensorRT INT8, no SAHI | ~30–40 |

---

### 4. Speed vs Precision Balance

**Chose precision over speed:**
- SAHI at 512×512 — costly but essential for tiny objects
- Confidence threshold at 0.2 — catches weak detections of small targets

**Chose speed over precision:**
- YOLO11n over larger variants — 3–4× faster, acceptable accuracy drop
- ByteTrack over DeepSORT — no Re-ID model needed, saves ~100 MB and inference time

Result: **~0.69 FPS on CPU** — not real-time, but produces high-quality tracked output. Near-real-time on Jetson with TensorRT.

---

### 5. Moving Camera Noise

Two types of noise from a moving camera:

- **Positional noise** (objects drift when camera pans) → handled by ego-motion compensation
- **Temporal noise** (flickering detections from motion blur) → handled by ByteTrack's 60-frame buffer and secondary matching of low-confidence detections

---

##  Performance

| Metric | Value |
|---|---|
| Model size | ~5.4 MB |
| Hardware | Intel Core i5-1035G1 @ 1.00GHz (CPU only) |
| Frames processed | 464 (~15 seconds) |
| Average FPS | 0.69 |
| Median FPS | 0.70 |
| Min FPS | 0.48 |
| Max FPS | 0.81 |
| Resolution | 1344×756 |
| SAHI patch size | 512×512, 20% overlap |

---

##  Installation

```bash
git clone <your-repo-url>
cd aerial-guardian
python -m venv .venv

# Windows
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

##  Running

**Main pipeline:**
```bash
python pipeline.py
```
Edit the bottom of `pipeline.py` to set your input/output paths:
```python
run_pipeline("your_video.mp4", "tracked_output.mp4")
```

**Baseline (optional):**
```bash
python baseline_detect.py --video your_video.mp4 --output baseline_output.mp4
```

---

##  Output Video

Each frame contains bounding boxes, unique ID labels, 50-frame trajectory tails, and a live FPS counter (green, top-left).
Dataset used : [Download it](https://drive.google.com/file/d/1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu/view?usp=sharing)
Output generated : [Take a look](https://drive.google.com/file/d/1-Q_fESn_bkiE3CBkbXNSJ_Z8I3IIBW1h/view?usp=drive_link)

---

##  Repository Structure

```
aerial-guardian/
├── pipeline.py          # Main pipeline
├── baseline_detect.py   # Baseline — plain YOLO11n, no SAHI, no tracking
├── stitch_video.py      # Converts VisDrone image sequences to MP4
├── requirements.txt
└── README.md
```

---

##  Requirements

```
ultralytics
opencv-python
sahi
supervision
```
