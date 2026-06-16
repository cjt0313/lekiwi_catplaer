# AprilTag Localization Design

## Goal

Detect an AprilTag (tagStandard41h12, 91mm) in each RGB frame to establish a fixed world reference frame, then transform all detection outputs (point cloud, segmentation mask points, 3D AABB) from camera frame into tag frame for visualization in viser.

## Architecture

Extend the existing `perception/viz_detection.py` with a `--localize` flag. Add a new `perception/localization.py` module that encapsulates AprilTag detection and pose estimation. The AprilTag defines the world origin — all geometry is rendered in tag coordinates when localization is active.

## Components

### `perception/localization.py`

Pure functions for AprilTag detection and coordinate transforms:

- `load_detector(family: str) -> apriltag.Detector` — Creates detector instance.
- `detect_tag_pose(detector, rgb: ndarray, K: ndarray, tag_size: float) -> Optional[ndarray]` — Converts RGB to grayscale, runs detection. If tag found, extracts pose and returns `T_camera_tag` as 4x4 homogeneous matrix. Returns None if no tag detected.
- `transform_points(points: ndarray, T: ndarray) -> ndarray` — Applies 4x4 rigid transform to Nx3 point array. Handles homogeneous conversion internally.
- `invert_transform(T: ndarray) -> ndarray` — Returns T^{-1} for a 4x4 rigid transform (uses R^T, -R^T @ t for efficiency).

### Changes to `perception/viz_detection.py`

New CLI arguments:
- `--localize` — Enable AprilTag-based frame transform (default: off, backward-compatible)

When `--localize` is enabled:
1. Load AprilTag detector at startup.
2. Each frame: detect AprilTag in the RGB image.
3. If tag detected: compute `T_tag_camera = inv(T_camera_tag)`. Transform all geometry (full point cloud, mask points, AABB corners, object center) into tag frame before rendering.
4. Render tag frame axes at viser origin (tag IS the world).
5. Render camera position/orientation as a frame marker in the tag coordinate system.
6. GUI text shows: tag detection status, object coordinates in tag frame.
7. If tag NOT detected in a frame: fall back to camera frame (same as current behavior). GUI shows "Tag lost — camera frame".

### Config additions (`common/config.py`)

```python
APRILTAG_FAMILY = "tagStandard41h12"
APRILTAG_SIZE = 0.091  # 91mm tag
```

## Coordinate Frame Convention

- `T_camera_tag`: 4x4 transform — tag pose in camera frame (what AprilTag detection returns)
- `T_tag_camera`: 4x4 transform — camera pose in tag frame (inverse of above)
- All rendered geometry uses tag frame as world when localization is active

## Dependencies

- `apriltag` — from `third_party/apriltag-pybind11`, compiled for Python 3.12 (gspy312 env)
- Existing: `pyorbbecsdk`, `ultralytics`, `viser`, `numpy`, `cv2`

## Build Requirement

The `apriltag-pybind11` submodule must be initialized and compiled for Python 3.12:
```bash
git submodule update --init third_party/apriltag-pybind11
cd third_party/apriltag-pybind11
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
# Copy .so to gspy312 site-packages
cp apriltag.cpython-312*.so $(conda run -n gspy312 python -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
```

## Testing

1. **`perception/test_apriltag.py`** — Single-frame test: capture RGB, detect tag, print T_camera_tag, save annotated image with tag corners and axes drawn. Verifies the library works with the real camera and 91mm tag.

2. **`python -m perception.viz_detection --localize --target cup`** — Full pipeline test: real-time viser visualization with tag at origin, cup detected and shown in tag coordinates. Verify that the point cloud is stable (tag frame is fixed), AABB coordinates make physical sense relative to tag placement.

## Success Criteria

- AprilTag detected reliably at 2-5 Hz with the 91mm tag
- Point cloud appears stable in viser (no jitter from camera motion if camera is stationary)
- Object AABB coordinates are in tag-relative meters and match physical measurement
- Graceful fallback when tag is occluded or out of frame
