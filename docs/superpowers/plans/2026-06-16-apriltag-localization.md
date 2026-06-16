# AprilTag Localization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add AprilTag-based localization to the detection visualization, transforming all 3D outputs into a fixed tag reference frame.

**Architecture:** New `perception/localization.py` module for AprilTag detection + pose. Extend `viz_detection.py` with `--localize` flag. AprilTag defines world origin.

**Tech Stack:** apriltag-pybind11 (Python 3.12 build), pyorbbecsdk, ultralytics, viser, numpy, opencv

---

### Task 1: Build and Install apriltag-pybind11 for Python 3.12

**Files:**
- Modify: `third_party/apriltag-pybind11/` (submodule init + build)

- [ ] **Step 1: Initialize the submodule**

```bash
cd /home/jim/code/VLA/lekiwi_catplaer/.claude/worktrees/init-claude-md
git submodule update --init third_party/apriltag-pybind11
```

- [ ] **Step 2: Build for Python 3.12**

```bash
cd third_party/apriltag-pybind11
mkdir -p build && cd build
conda run -n gspy312 cmake -DCMAKE_BUILD_TYPE=Release ..
conda run -n gspy312 cmake --build . -j$(nproc)
```

Expected: Produces `apriltag.cpython-312-x86_64-linux-gnu.so` in `build/`

- [ ] **Step 3: Install the .so to gspy312 site-packages**

```bash
PLATLIB=$(conda run -n gspy312 python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
cp third_party/apriltag-pybind11/build/apriltag.cpython-312*.so "$PLATLIB/"
```

- [ ] **Step 4: Verify import**

```bash
conda run -n gspy312 python -c "import apriltag; d = apriltag.Detector(family='tagStandard41h12'); print('OK:', d)"
```

Expected: No error, prints detector object.

- [ ] **Step 5: Commit**

No code change to commit (binary in site-packages, submodule already tracked).

---

### Task 2: Create `perception/localization.py`

**Files:**
- Create: `perception/localization.py`

- [ ] **Step 1: Write the module**

```python
"""AprilTag detection and pose estimation."""

import cv2
import numpy as np
import apriltag


def load_detector(family="tagStandard41h12"):
    return apriltag.Detector(family=family)


def detect_tag_pose(detector, rgb, K, tag_size=0.091):
    """Detect AprilTag and return T_camera_tag (4x4) or None."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detections = detector.detect(gray)

    if len(detections) == 0:
        return None

    detection = detections[0]
    pose_res = detection.pose(
        tagsize=tag_size,
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2],
    )

    T_camera_tag = np.eye(4)
    T_camera_tag[:3, :3] = pose_res.R
    T_camera_tag[:3, 3] = pose_res.t.flatten()
    return T_camera_tag


def invert_transform(T):
    """Invert a 4x4 rigid transform efficiently."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(points, T):
    """Apply 4x4 rigid transform to Nx3 points. Returns Nx3."""
    N = len(points)
    if N == 0:
        return points
    ones = np.ones((N, 1), dtype=points.dtype)
    pts_h = np.hstack([points, ones])  # Nx4
    transformed = (T @ pts_h.T).T  # Nx4
    return transformed[:, :3]
```

- [ ] **Step 2: Verify module imports**

```bash
conda run -n gspy312 python -c "from perception.localization import load_detector, detect_tag_pose, invert_transform, transform_points; print('OK')"
```

Run from repo root. Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add perception/localization.py
git commit -m "feat: add perception/localization.py for AprilTag pose estimation"
```

---

### Task 3: Create `perception/test_apriltag.py`

**Files:**
- Create: `perception/test_apriltag.py`

- [ ] **Step 1: Write the test script**

```python
"""Test AprilTag detection on live camera. Saves annotated image with tag pose info."""

import sys
import cv2
import numpy as np

sys.path.insert(0, ".")
from common.config import APRILTAG_FAMILY, APRILTAG_SIZE
from perception.camera import OrbbecCamera
from perception.localization import load_detector, detect_tag_pose, invert_transform


def draw_tag_axes(image, K, T_camera_tag, axis_length=0.05):
    """Draw 3D axes on the tag center projected into the image."""
    origin = T_camera_tag[:3, 3]
    axes_pts = np.float32([
        origin,
        origin + T_camera_tag[:3, 0] * axis_length,
        origin + T_camera_tag[:3, 1] * axis_length,
        origin + T_camera_tag[:3, 2] * axis_length,
    ])

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    proj = np.zeros((4, 2), dtype=np.int32)
    for i, pt in enumerate(axes_pts):
        proj[i, 0] = int(pt[0] * fx / pt[2] + cx)
        proj[i, 1] = int(pt[1] * fy / pt[2] + cy)

    cv2.line(image, tuple(proj[0]), tuple(proj[1]), (255, 0, 0), 2)  # X red
    cv2.line(image, tuple(proj[0]), tuple(proj[2]), (0, 255, 0), 2)  # Y green
    cv2.line(image, tuple(proj[0]), tuple(proj[3]), (0, 0, 255), 2)  # Z blue
    return image


def main():
    print(f"AprilTag family: {APRILTAG_FAMILY}, size: {APRILTAG_SIZE}m")
    detector = load_detector(APRILTAG_FAMILY)

    print("Starting camera...")
    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    for _ in range(5):
        camera.grab()

    print("Capturing frame...")
    rgb, depth = camera.grab()

    if rgb is None:
        print("ERROR: Failed to capture frame")
        camera.stop()
        return

    T_camera_tag = detect_tag_pose(detector, rgb, K, tag_size=APRILTAG_SIZE)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if T_camera_tag is None:
        print("No AprilTag detected!")
        cv2.putText(bgr, "No tag detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite("test_output_apriltag.png", bgr)
        camera.stop()
        return

    T_tag_camera = invert_transform(T_camera_tag)
    tag_pos = T_camera_tag[:3, 3]

    print(f"\nT_camera_tag (tag pose in camera frame):")
    print(f"  position: [{tag_pos[0]:.4f}, {tag_pos[1]:.4f}, {tag_pos[2]:.4f}] m")
    print(f"  distance: {np.linalg.norm(tag_pos):.4f} m")
    print(f"\nFull T_camera_tag:\n{T_camera_tag}")

    bgr = draw_tag_axes(bgr, K, T_camera_tag)
    info_lines = [
        f"Tag detected",
        f"pos: ({tag_pos[0]:.3f}, {tag_pos[1]:.3f}, {tag_pos[2]:.3f})m",
        f"dist: {np.linalg.norm(tag_pos):.3f}m",
    ]
    y_offset = 30
    for line in info_lines:
        cv2.putText(bgr, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    cv2.imwrite("test_output_apriltag.png", bgr)
    print(f"\nSaved: test_output_apriltag.png")

    camera.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test with real camera**

```bash
conda run -n gspy312 python -m perception.test_apriltag
```

Expected: Detects tag, prints position (should be ~0.3-1.0m distance), saves annotated image showing RGB/green/blue axes on tag.

- [ ] **Step 3: Commit**

```bash
git add perception/test_apriltag.py
git commit -m "feat: add AprilTag detection test script"
```

---

### Task 4: Add config constants

**Files:**
- Modify: `common/config.py` (append 2 lines)

- [ ] **Step 1: Add AprilTag constants to config**

Append to `common/config.py`:

```python
# AprilTag localization
APRILTAG_FAMILY = "tagStandard41h12"
APRILTAG_SIZE = 0.091  # 91mm tag
```

- [ ] **Step 2: Commit**

```bash
git add common/config.py
git commit -m "feat: add AprilTag config constants"
```

---

### Task 5: Extend `viz_detection.py` with `--localize` flag

**Files:**
- Modify: `perception/viz_detection.py`

- [ ] **Step 1: Add imports for localization**

Add to the imports section of `viz_detection.py`:

```python
from common.config import (
    FAKE_CAMERA_PUB, DETECTION_TARGET_CLASS,
    DETECTION_CONF_THRESHOLD, DETECTION_MODEL, DEPTH_SCALE,
    APRILTAG_FAMILY, APRILTAG_SIZE,
)
from perception.localization import (
    load_detector, detect_tag_pose, invert_transform, transform_points,
)
```

- [ ] **Step 2: Add `--localize` argument**

In the `main()` argparse section, add:

```python
parser.add_argument("--localize", action="store_true",
                    help="Enable AprilTag-based frame transform (tag = world origin)")
```

- [ ] **Step 3: Initialize AprilTag detector when --localize is set**

After camera initialization, add:

```python
tag_detector = None
if args.localize:
    print(f"Loading AprilTag detector: {APRILTAG_FAMILY}, size={APRILTAG_SIZE}m")
    tag_detector = load_detector(APRILTAG_FAMILY)
    server.scene.add_frame("/tag_origin", axes_length=0.1, axes_radius=0.004)
```

- [ ] **Step 4: Add AprilTag detection + transform in the main loop**

After detection logic and before viser scene updates, add the frame transform block:

```python
# Frame transform (localization)
T_tag_camera = None
if tag_detector is not None:
    T_camera_tag = detect_tag_pose(tag_detector, rgb, K, tag_size=APRILTAG_SIZE)
    if T_camera_tag is not None:
        T_tag_camera = invert_transform(T_camera_tag)
        # Transform full point cloud
        points_full = transform_points(points_full, T_tag_camera)
        # Transform detection points
        if detected:
            obj_points = transform_points(obj_points, T_tag_camera)
            center, bbox_min, bbox_max = fit_aabb(obj_points)
            size = bbox_max - bbox_min
```

- [ ] **Step 5: Add camera frame visualization when localized**

After the transform block, add camera position indicator:

```python
if T_tag_camera is not None:
    cam_pos = T_tag_camera[:3, 3]
    server.scene.add_frame(
        "/scene/camera_pose",
        position=tuple(cam_pos),
        wxyz=tuple(rotation_matrix_to_quaternion(T_tag_camera[:3, :3])),
        axes_length=0.05,
        axes_radius=0.002,
    )
    gui_status_prefix = f"[TAG] "
else:
    gui_status_prefix = "[CAM] "
    if tag_detector is not None:
        gui_status_prefix = "[TAG LOST - CAM] "
```

- [ ] **Step 6: Add quaternion helper function**

Add this utility function near the top of the file (after imports):

```python
def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to (w, x, y, z) quaternion for viser."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (w, x, y, z)
```

- [ ] **Step 7: Update GUI status text to include frame prefix**

Modify the existing `gui_status.value = ...` lines to prepend `gui_status_prefix`:

For the detected case:
```python
gui_status.value = (
    f"{gui_status_prefix}DETECTED conf={confidence:.2f} "
    f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
    f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})"
)
```

For the no-detection case:
```python
gui_status.value = f"{gui_status_prefix}No detection"
```

- [ ] **Step 8: Run the full localized visualization**

```bash
conda run -n gspy312 python -m perception.viz_detection --localize --target cup
```

Open http://localhost:8080. Expected:
- Tag axes visible at origin (0,0,0)
- Point cloud rendered in tag frame (stable, no jitter if camera is static)
- Cup detected with green mask points + yellow AABB wireframe
- GUI shows `[TAG] DETECTED conf=0.XX center=(x,y,z) size=(w,h,d)`
- Coordinates in tag frame make physical sense (e.g., cup is 0.3m from tag)

- [ ] **Step 9: Commit**

```bash
git add perception/viz_detection.py
git commit -m "feat: add --localize flag for AprilTag frame transform in viz_detection"
```

---

### Task 6: Add `.gitignore` entries

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add test output images**

Append to `.gitignore`:

```
test_output_apriltag.png
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore apriltag test output"
```
