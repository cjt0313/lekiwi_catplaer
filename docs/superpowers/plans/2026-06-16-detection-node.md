# Detection Node Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real camera detection node that captures RGB-D from an Orbbec camera, runs YOLO-seg to detect objects, back-projects to 3D, and publishes a 3D bounding box on ZMQ — tested incrementally with visual output at each stage.

**Architecture:** `perception/camera.py` (Orbbec capture) → `perception/detection.py` (YOLO + 3D projection) → `nodes/real_camera_node.py` (orchestrator publishing on ZMQ). Each module is tested standalone with a visualization script before integration.

**Tech Stack:** pyorbbecsdk 2.x, ultralytics (yolo11x-seg), numpy, opencv-python, pyzmq

---

## File Map

| File | Role |
|------|------|
| `perception/__init__.py` | Package marker |
| `perception/camera.py` | Orbbec pyorbbecsdk wrapper: start/stop, grab aligned RGB-D, expose intrinsics |
| `perception/detection.py` | YOLO-seg model loading, mask extraction, depth→3D backprojection, outlier removal, AABB fitting |
| `perception/test_camera.py` | Standalone camera test — saves RGB + colorized depth images |
| `perception/test_detection.py` | Standalone detection test — saves annotated image with mask + 3D bbox overlay |
| `nodes/real_camera_node.py` | Thin orchestrator: capture → detect → publish CAT_BBOX_3D on ZMQ |
| `common/config.py` | Add detection config constants |
| `requirements.txt` | Add ultralytics, opencv-python, pyorbbecsdk |

---

### Task 1: Install Dependencies

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

Add these lines to the end of `requirements.txt`:

```
ultralytics>=8.0
opencv-python>=4.8
```

Note: `pyorbbecsdk` is already installed system-wide (v2.0.13). Don't add it to requirements.txt as it's installed via pip from Orbbec's wheel, not PyPI.

- [ ] **Step 2: Install ultralytics**

Run:
```bash
conda activate lekiwi && pip install ultralytics opencv-python
```

Expected: successful install, `ultralytics` importable.

- [ ] **Step 3: Verify YOLO model downloads**

Run:
```bash
python -c "from ultralytics import YOLO; model = YOLO('yolo11x-seg.pt'); print('Model loaded:', model.model.names[15])"
```

Expected: downloads `yolo11x-seg.pt` (~130MB), prints `Model loaded: cat`. Class 15 is "cat" in COCO, class 41 is "cup".

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "feat: add ultralytics and opencv to requirements"
```

---

### Task 2: Orbbec Camera Wrapper (`perception/camera.py`)

**Files:**
- Create: `perception/__init__.py`
- Create: `perception/camera.py`

- [ ] **Step 1: Create perception package**

Create `perception/__init__.py`:

```python
```

(Empty file — package marker only.)

- [ ] **Step 2: Write `perception/camera.py`**

```python
"""Orbbec RGB-D camera wrapper using pyorbbecsdk."""

import numpy as np
from pyorbbecsdk import (
    Pipeline, Config, OBStreamType, OBFormat, OBAlignMode,
)


class OrbbecCamera:
    def __init__(self, color_width=640, color_height=480, color_fps=30,
                 depth_width=640, depth_height=480, depth_fps=30):
        self._color_w = color_width
        self._color_h = color_height
        self._color_fps = color_fps
        self._depth_w = depth_width
        self._depth_h = depth_height
        self._depth_fps = depth_fps
        self._pipeline = None
        self._intrinsics = None

    def start(self):
        self._pipeline = Pipeline()

        config = Config()

        color_profiles = self._pipeline.get_stream_profile_list(OBStreamType.COLOR_STREAM)
        color_profile = color_profiles.get_video_stream_profile(
            self._color_w, self._color_h, OBFormat.RGB, self._color_fps
        )
        config.enable_stream(color_profile)

        depth_profiles = self._pipeline.get_stream_profile_list(OBStreamType.DEPTH_STREAM)
        depth_profile = depth_profiles.get_video_stream_profile(
            self._depth_w, self._depth_h, OBFormat.Y16, self._depth_fps
        )
        config.enable_stream(depth_profile)

        config.set_align_mode(OBAlignMode.HW_MODE)

        self._pipeline.start(config)

        # Read intrinsics from the depth stream profile
        intrinsic = depth_profile.get_intrinsic()
        self._intrinsics = np.array([
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        print(f"[OrbbecCamera] Started: color={self._color_w}x{self._color_h}@{self._color_fps}fps, "
              f"depth={self._depth_w}x{self._depth_h}@{self._depth_fps}fps")
        print(f"[OrbbecCamera] Intrinsics: fx={intrinsic.fx:.1f} fy={intrinsic.fy:.1f} "
              f"cx={intrinsic.cx:.1f} cy={intrinsic.cy:.1f}")

    def grab(self, timeout_ms=1000):
        """Grab aligned RGB-D frame pair.

        Returns:
            (rgb, depth) where rgb is HxWx3 uint8, depth is HxW uint16 (mm).
            Returns (None, None) if timeout.
        """
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        rgb = np.asarray(color_frame.get_data()).reshape(
            (color_frame.get_height(), color_frame.get_width(), 3)
        )

        depth = np.asarray(depth_frame.get_data()).reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        ).astype(np.uint16)

        return rgb, depth

    @property
    def intrinsics(self):
        return self._intrinsics

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            print("[OrbbecCamera] Stopped")
```

- [ ] **Step 3: Commit**

```bash
git add perception/__init__.py perception/camera.py
git commit -m "feat: add Orbbec camera wrapper (perception/camera.py)"
```

---

### Task 3: Test Camera Capture with Visualization

**Files:**
- Create: `perception/test_camera.py`

- [ ] **Step 1: Write `perception/test_camera.py`**

```python
"""Test Orbbec camera capture. Saves sample RGB + depth images for visual verification."""

import sys
import cv2
import numpy as np

sys.path.insert(0, ".")
from perception.camera import OrbbecCamera


def colorize_depth(depth, max_depth_mm=5000):
    """Convert 16-bit depth to a colorized visualization."""
    valid = depth > 0
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if valid.any():
        d = depth.astype(np.float32)
        d[~valid] = 0
        d = np.clip(d / max_depth_mm, 0, 1)
        normalized = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def main():
    camera = OrbbecCamera()
    camera.start()

    print("Capturing 5 frames (skipping first 3 for warmup)...")
    for i in range(5):
        rgb, depth = camera.grab()
        if rgb is None:
            print(f"  Frame {i}: timeout")
            continue
        print(f"  Frame {i}: rgb={rgb.shape} depth={depth.shape} "
              f"depth_range=[{depth[depth > 0].min() if (depth > 0).any() else 0}, "
              f"{depth.max()}] mm")

    # Save the last good frame
    if rgb is not None and depth is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_vis = colorize_depth(depth)

        cv2.imwrite("test_output_rgb.png", bgr)
        cv2.imwrite("test_output_depth.png", depth_vis)

        # Side-by-side composite
        composite = np.hstack([bgr, depth_vis])
        cv2.imwrite("test_output_camera.png", composite)
        print(f"\nSaved: test_output_camera.png ({composite.shape[1]}x{composite.shape[0]})")
        print(f"  Left: RGB image")
        print(f"  Right: Depth colormap (blue=near, red=far)")
        print(f"\nIntrinsics K:\n{camera.intrinsics}")
    else:
        print("ERROR: No frames captured!")

    camera.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the camera test**

Run:
```bash
python -m perception.test_camera
```

Expected output:
- Prints frame shapes and depth ranges
- Saves `test_output_camera.png` — side-by-side RGB and colorized depth
- Prints intrinsics matrix

Verify: Open `test_output_camera.png`. Left half should show the scene in color, right half should show depth as a JET colormap (blue=near, red=far). Objects closer to the camera should appear bluer.

- [ ] **Step 3: Commit**

```bash
git add perception/test_camera.py
git commit -m "feat: add camera test script with RGB+depth visualization"
```

---

### Task 4: Detection Module (`perception/detection.py`)

**Files:**
- Create: `perception/detection.py`

- [ ] **Step 1: Write `perception/detection.py`**

```python
"""YOLO-seg detection + depth backprojection + 3D AABB."""

import numpy as np
import cv2
from ultralytics import YOLO


def load_model(weights="yolo11x-seg.pt"):
    """Load YOLO segmentation model. Downloads weights on first call."""
    model = YOLO(weights)
    return model


def detect_target(model, rgb, target_class, conf_threshold=0.25):
    """Run YOLO-seg and extract the highest-confidence mask for target_class.

    Args:
        model: YOLO model instance
        rgb: HxWx3 uint8 image (RGB order)
        target_class: COCO class name, e.g. "cat" or "cup"
        conf_threshold: minimum detection confidence

    Returns:
        (mask, confidence) where mask is HxW bool, or None if not detected.
    """
    results = model.predict(source=rgb, conf=conf_threshold, retina_masks=True, verbose=False)
    result = results[0]

    if result.masks is None or result.boxes is None:
        return None

    H, W = rgb.shape[:2]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    best_conf = -1.0
    best_mask = None

    for i, cls_id in enumerate(class_ids):
        class_name = result.names[cls_id]
        if class_name == target_class and confs[i] > best_conf:
            best_conf = confs[i]
            best_mask = masks[i]

    if best_mask is None:
        return None

    if best_mask.shape != (H, W):
        best_mask = cv2.resize(best_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    return best_mask > 0.5, float(best_conf)


def mask_to_3d_points(mask, depth, K, depth_scale=1000.0):
    """Back-project masked depth pixels to 3D points in camera frame.

    Args:
        mask: HxW boolean mask
        depth: HxW uint16 depth image
        K: 3x3 camera intrinsic matrix
        depth_scale: divisor to convert depth to meters (1000.0 for mm)

    Returns:
        Nx3 float32 array of 3D points in camera frame (meters).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.where(mask)
    zs = depth[ys, xs].astype(np.float32) / depth_scale

    valid = zs > 0
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.stack([X, Y, Z], axis=1)


def remove_outliers(points, q_low=0.05, q_high=0.95):
    """Remove outlier points using per-axis quantile filtering."""
    if len(points) < 10:
        return points

    lo = np.quantile(points, q_low, axis=0)
    hi = np.quantile(points, q_high, axis=0)

    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def fit_aabb(points):
    """Fit axis-aligned 3D bounding box to point cloud.

    Returns:
        (center, bbox_min, bbox_max) — each a 3-element numpy array (meters).
    """
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    return center, bbox_min, bbox_max
```

- [ ] **Step 2: Commit**

```bash
git add perception/detection.py
git commit -m "feat: add YOLO-seg detection + 3D backprojection module"
```

---

### Task 5: Test Detection with Cup Visualization

**Files:**
- Create: `perception/test_detection.py`

- [ ] **Step 1: Write `perception/test_detection.py`**

```python
"""Test YOLO detection on live camera. Saves annotated image with mask + 3D bbox info."""

import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, ".")
from perception.camera import OrbbecCamera
from perception.detection import load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb


def draw_mask_overlay(bgr, mask, color=(0, 255, 0), alpha=0.4):
    """Draw semi-transparent mask overlay on image."""
    overlay = bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)


def draw_bbox_2d(image, mask, color=(0, 255, 0)):
    """Draw 2D bounding box around the mask region."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="cup", help="COCO class name to detect")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--weights", default="yolo11x-seg.pt", help="YOLO model weights")
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.weights}")
    model = load_model(args.weights)

    print("Starting camera...")
    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    # Warm up: skip first few frames
    for _ in range(5):
        camera.grab()

    print(f"Detecting '{args.target}' (conf >= {args.conf})...")
    rgb, depth = camera.grab()

    if rgb is None:
        print("ERROR: Failed to capture frame")
        camera.stop()
        return

    result = detect_target(model, rgb, args.target, args.conf)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if result is None:
        print(f"'{args.target}' NOT DETECTED. Saving raw image.")
        cv2.putText(bgr, f"No '{args.target}' detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite("test_output_detection.png", bgr)
        camera.stop()
        return

    mask, confidence = result
    print(f"Detected '{args.target}' with confidence={confidence:.3f}")
    print(f"  Mask pixels: {mask.sum()}")

    # 3D bbox computation
    points = mask_to_3d_points(mask, depth, K, depth_scale=1000.0)
    print(f"  3D points (before outlier removal): {len(points)}")

    if len(points) < 10:
        print("  WARNING: Too few valid depth points!")
        camera.stop()
        return

    points_clean = remove_outliers(points)
    print(f"  3D points (after outlier removal): {len(points_clean)}")

    center, bbox_min, bbox_max = fit_aabb(points_clean)
    size = bbox_max - bbox_min

    print(f"\n  3D Bounding Box (camera frame, meters):")
    print(f"    center:   [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"    bbox_min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
    print(f"    bbox_max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
    print(f"    size:     [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")

    # Draw visualization
    vis = draw_mask_overlay(bgr, mask, color=(0, 255, 100))
    vis = draw_bbox_2d(vis, mask, color=(0, 255, 0))

    # Add text annotations
    info_lines = [
        f"{args.target} conf={confidence:.2f}",
        f"center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})m",
        f"size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})m",
        f"depth: {center[2]:.2f}m",
    ]
    y_offset = 30
    for line in info_lines:
        cv2.putText(vis, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    cv2.imwrite("test_output_detection.png", vis)
    print(f"\nSaved: test_output_detection.png")
    print(f"  Shows: RGB with green mask overlay, 2D bbox, 3D info text")

    camera.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run detection test with a cup**

Place a cup in front of the camera, then run:

```bash
python -m perception.test_detection --target cup
```

Expected output:
- Prints confidence, mask pixel count, 3D point counts
- Prints 3D bounding box center and size in meters
- Saves `test_output_detection.png` with green mask overlay, 2D bounding box, and 3D bbox text

Verify: Open `test_output_detection.png`. The cup should have a green semi-transparent overlay. The depth value should be reasonable (e.g., center Z ~ 0.3-1.5m). The size should roughly match the physical cup dimensions (e.g., ~0.08m width, ~0.12m height).

- [ ] **Step 3: Test with different objects (optional)**

Try other COCO classes to verify flexibility:
```bash
python -m perception.test_detection --target bottle
python -m perception.test_detection --target cat
```

- [ ] **Step 4: Commit**

```bash
git add perception/test_detection.py
git commit -m "feat: add detection test script with mask + 3D bbox visualization"
```

---

### Task 6: Config Additions

**Files:**
- Modify: `common/config.py`

- [ ] **Step 1: Add detection config to `common/config.py`**

Append these lines at the end of `common/config.py`:

```python
# Detection parameters
DETECTION_TARGET_CLASS = "cat"
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_MODEL = "yolo11x-seg.pt"
DEPTH_SCALE = 1000.0  # Orbbec depth in millimeters
```

- [ ] **Step 2: Commit**

```bash
git add common/config.py
git commit -m "feat: add detection config parameters"
```

---

### Task 7: Real Camera Node (`nodes/real_camera_node.py`)

**Files:**
- Create: `nodes/real_camera_node.py`

- [ ] **Step 1: Write `nodes/real_camera_node.py`**

```python
"""Real camera node: captures RGB-D from Orbbec, detects target, publishes CAT_BBOX_3D."""

import sys
import time

sys.path.insert(0, ".")

from common.config import (
    FAKE_CAMERA_PUB, DETECTION_TARGET_CLASS,
    DETECTION_CONF_THRESHOLD, DETECTION_MODEL, DEPTH_SCALE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, publish, ZmqMessage
from perception.camera import OrbbecCamera
from perception.detection import load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb


class RealCameraNode:
    def __init__(self):
        self.pub = make_publisher(FAKE_CAMERA_PUB)
        self.camera = OrbbecCamera()
        self.model = None
        self.seq = 0

    def start(self):
        print(f"[RealCamera] Loading model: {DETECTION_MODEL}")
        self.model = load_model(DETECTION_MODEL)

        print("[RealCamera] Starting camera...")
        self.camera.start()

        # Warm up: skip first few frames
        for _ in range(5):
            self.camera.grab()

        print("[RealCamera] Ready. Publishing CAT_BBOX_3D on", FAKE_CAMERA_PUB)

    def step(self):
        rgb, depth = self.camera.grab()
        if rgb is None:
            return

        K = self.camera.intrinsics
        result = detect_target(self.model, rgb, DETECTION_TARGET_CLASS, DETECTION_CONF_THRESHOLD)

        if result is not None:
            mask, confidence = result
            points = mask_to_3d_points(mask, depth, K, depth_scale=DEPTH_SCALE)
            points = remove_outliers(points)

            if len(points) >= 10:
                center, bbox_min, bbox_max = fit_aabb(points)
                payload = {
                    "center": center.tolist(),
                    "bbox_min": bbox_min.tolist(),
                    "bbox_max": bbox_max.tolist(),
                    "confidence": confidence,
                    "visible": True,
                }
            else:
                payload = {
                    "center": None,
                    "bbox_min": None,
                    "bbox_max": None,
                    "confidence": 0.0,
                    "visible": False,
                }
        else:
            payload = {
                "center": None,
                "bbox_min": None,
                "bbox_max": None,
                "confidence": 0.0,
                "visible": False,
            }

        msg = ZmqMessage.create(MsgType.CAT_BBOX_3D, payload, "real_camera_node",
                                seq=self.seq, frame_id="camera")
        publish(self.pub, msg)
        self.seq += 1

        if self.seq % 10 == 0:
            if payload["visible"]:
                c = payload["center"]
                print(f"[RealCamera] seq={self.seq} DETECTED center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                      f"conf={payload['confidence']:.2f}")
            else:
                print(f"[RealCamera] seq={self.seq} no detection")

    def run(self):
        self.start()
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            print("\n[RealCamera] Shutting down...")
        finally:
            self.camera.stop()


def main():
    node = RealCameraNode()
    node.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the real camera node**

```bash
python -m nodes.real_camera_node
```

Expected: prints detection results every 10 frames (center coordinates when detected, "no detection" otherwise). Ctrl+C to stop cleanly.

- [ ] **Step 3: Verify ZMQ messages with a subscriber**

In a separate terminal, run a quick subscriber to see published messages:

```bash
python -c "
import sys; sys.path.insert(0, '.')
from common.config import FAKE_CAMERA_PUB
from common.zmq_message import make_subscriber, receive
from common.types import MsgType
sub = make_subscriber(FAKE_CAMERA_PUB, [MsgType.CAT_BBOX_3D.value])
print('Listening for CAT_BBOX_3D...')
while True:
    msg = receive(sub, timeout_ms=2000)
    if msg:
        print(f'  visible={msg.payload[\"visible\"]} center={msg.payload[\"center\"]}')
"
```

Expected: prints detection payloads as they arrive. With a cat (or target object) visible, `visible=True` and center coordinates in meters. Without, `visible=False`.

- [ ] **Step 4: Commit**

```bash
git add nodes/real_camera_node.py
git commit -m "feat: add real_camera_node with YOLO-seg 3D detection"
```

---

### Task 8: Full Pipeline Visualization Test

**Files:**
- Create: `perception/test_full_pipeline.py`

- [ ] **Step 1: Write `perception/test_full_pipeline.py`**

This script runs the full pipeline continuously and produces a live-updating visualization image (saved every N frames):

```python
"""Full pipeline test: continuous detection with live visualization output."""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, ".")
from perception.camera import OrbbecCamera
from perception.detection import load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb


def colorize_depth(depth, max_depth_mm=3000):
    valid = depth > 0
    d = depth.astype(np.float32)
    d[~valid] = 0
    d = np.clip(d / max_depth_mm, 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def main():
    target_class = sys.argv[1] if len(sys.argv) > 1 else "cup"
    weights = sys.argv[2] if len(sys.argv) > 2 else "yolo11x-seg.pt"

    print(f"Full pipeline test: target='{target_class}', model={weights}")
    model = load_model(weights)

    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    # Warm up
    for _ in range(5):
        camera.grab()

    frame_count = 0
    fps_start = time.time()
    save_interval = 15  # save visualization every N frames

    print("Running... Press Ctrl+C to stop. Visualization saved periodically.")

    try:
        while True:
            rgb, depth = camera.grab()
            if rgb is None:
                continue

            t0 = time.time()
            result = detect_target(model, rgb, target_class, conf_threshold=0.25)
            inference_ms = (time.time() - t0) * 1000

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_vis = colorize_depth(depth)

            if result is not None:
                mask, confidence = result
                points = mask_to_3d_points(mask, depth, K, depth_scale=1000.0)
                points = remove_outliers(points)

                # Draw mask overlay
                overlay = bgr.copy()
                overlay[mask] = (0, 255, 100)
                bgr = cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0)

                if len(points) >= 10:
                    center, bbox_min, bbox_max = fit_aabb(points)
                    size = bbox_max - bbox_min

                    # Draw 2D bbox
                    ys, xs = np.where(mask)
                    cv2.rectangle(bgr, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 2)

                    # Info text
                    lines = [
                        f"{target_class} {confidence:.2f}",
                        f"Z={center[2]:.2f}m",
                        f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})",
                    ]
                    for i, line in enumerate(lines):
                        cv2.putText(bgr, line, (xs.min(), ys.min() - 10 - i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Also draw depth points on depth vis
                    depth_vis[mask] = (0, 255, 0)

                    status = f"DETECTED: center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
                else:
                    status = "DETECTED but too few depth points"
            else:
                status = "No detection"

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(bgr, f"FPS: {fps:.1f}  Inference: {inference_ms:.0f}ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, status, (10, bgr.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if frame_count % save_interval == 0:
                composite = np.hstack([bgr, depth_vis])
                cv2.imwrite("test_output_pipeline.png", composite)

            if frame_count % 30 == 0:
                print(f"  frame={frame_count} fps={fps:.1f} inference={inference_ms:.0f}ms {status}")

    except KeyboardInterrupt:
        pass

    # Save final frame
    composite = np.hstack([bgr, depth_vis])
    cv2.imwrite("test_output_pipeline.png", composite)
    print(f"\nFinal visualization saved: test_output_pipeline.png")
    print(f"Total frames: {frame_count}, avg FPS: {frame_count / elapsed:.1f}")

    camera.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline test**

Place a cup in front of the camera:

```bash
python -m perception.test_full_pipeline cup
```

Expected:
- Prints FPS, inference time, detection status every 30 frames
- Saves `test_output_pipeline.png` periodically (side-by-side RGB+depth with annotations)
- On RTX 4060 with yolo11x-seg, expect 10-20 FPS

Verify: Open `test_output_pipeline.png`. Should show:
- Left: RGB with green mask overlay on the cup, 2D bounding box, confidence, depth, and 3D size text
- Right: Depth colormap with mask highlighted in green
- FPS counter in top-left

- [ ] **Step 3: Commit**

```bash
git add perception/test_full_pipeline.py
git commit -m "feat: add full pipeline visualization test"
```

---

### Task 9: Add .gitignore for Test Outputs

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add test output files to .gitignore**

Append to `.gitignore`:

```
# Test visualization outputs
test_output_*.png
*.pt
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore test output images and model weights"
```

---

## Summary

After completing all tasks, the detection pipeline is fully functional:

1. `python -m perception.test_camera` — verifies camera captures RGB-D correctly
2. `python -m perception.test_detection --target cup` — verifies YOLO detects a cup and produces valid 3D bbox
3. `python -m perception.test_full_pipeline cup` — continuous detection with live visualization
4. `python -m nodes.real_camera_node` — production node publishing on ZMQ for downstream consumers
