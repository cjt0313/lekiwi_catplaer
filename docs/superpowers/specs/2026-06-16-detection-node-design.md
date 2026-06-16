# Detection Node Design

Real camera node with YOLO-seg 3D object detection using Orbbec RGB-D camera.

## Goal

Replace the fake cat detection (hardcoded position in `fake_camera_node`) with real detection: capture RGB-D frames from an Orbbec camera, run YOLO instance segmentation to get a 2D mask, back-project masked depth pixels to 3D, and publish a 3D axis-aligned bounding box on the existing `CAT_BBOX_3D` ZMQ topic.

This is phase 1 of incrementally replacing `fake_camera_node`. Detection first; localization and mapping come later as separate additions to the same `real_camera_node`.

## Architecture

```
perception/
  __init__.py
  camera.py            # Orbbec pyorbbecsdk wrapper
  detection.py         # YOLO-seg + depth backprojection + 3D AABB
nodes/
  real_camera_node.py  # Orchestrator: capture → detect → publish
```

### `perception/camera.py`

Thin wrapper around pyorbbecsdk. Responsibilities:

- Open device, configure color stream (640×480 RGB) and depth stream (640×480, 16-bit mm)
- Enable hardware depth-to-color (D2C) alignment so depth pixels correspond 1:1 with color pixels
- Expose `grab() → (rgb: HxWx3 uint8, depth: HxW uint16)` — returns the latest aligned frame pair
- Expose `intrinsics` property → 3×3 numpy intrinsic matrix K (read from depth stream profile at startup)
- `start()` / `stop()` lifecycle

No detection logic here. This module is reusable by future localization and mapping code.

### `perception/detection.py`

Stateless functions (plus model loading):

| Function | Input | Output |
|----------|-------|--------|
| `load_model(weights)` | path to `.pt` file | `YOLO` model object |
| `detect_target(model, rgb, target_class, conf)` | model, HxWx3 image, class name, threshold | `(mask: HxW bool, confidence: float)` or `None` |
| `mask_to_3d_points(mask, depth, K, depth_scale)` | binary mask, depth image, intrinsics, scale | `Nx3 float32` points in camera frame |
| `remove_outliers(points, q_low, q_high)` | point cloud, quantile bounds | filtered `Nx3` array |
| `fit_aabb(points)` | `Nx3` array | `(center, bbox_min, bbox_max)` each 3-element arrays |

Pipeline: `detect_target` → `mask_to_3d_points` → `remove_outliers` → `fit_aabb`.

YOLO model: `yolo11x-seg.pt` (pretrained COCO, x-size for best accuracy; RTX 4060 has sufficient VRAM). Loaded once at startup. Inference with `retina_masks=True` for full-resolution masks.

### `nodes/real_camera_node.py`

Thin orchestrator. Main loop:

```
initialize camera, model, ZMQ publisher
while True:
    rgb, depth = camera.grab()
    result = detect_target(model, rgb, target_class, conf)
    if result:
        mask, confidence = result
        points = mask_to_3d_points(mask, depth, K, depth_scale)
        points = remove_outliers(points)
        if len(points) >= 10:
            center, bbox_min, bbox_max = fit_aabb(points)
            publish CAT_BBOX_3D with visible=True
        else:
            publish CAT_BBOX_3D with visible=False
    else:
        publish CAT_BBOX_3D with visible=False
```

Publishes on `FAKE_CAMERA_PUB` (tcp://127.0.0.1:5560) — same address as fake_camera_node so downstream nodes subscribe unchanged. Only the `CAT_BBOX_3D` topic is published for now; `ROBOT_POSE` and `OCCUPANCY_GRID` will be added when localization/mapping are implemented.

Loop rate: as fast as camera + inference allows (expected 5-15 Hz with yolov8n-seg on GPU, 2-5 Hz on CPU).

## Message Format

Published on topic `cat_bbox_3d`:

```json
{
    "center": [0.42, 0.15, 1.2],
    "bbox_min": [0.30, 0.0, 1.05],
    "bbox_max": [0.54, 0.30, 1.35],
    "confidence": 0.87,
    "visible": true
}
```

When no detection: `visible: false`, other fields set to `null`.

**Coordinate frame**: Camera optical frame (Z forward, X right, Y down). The `frame_id` in the ZMQ header is set to `"camera"` (vs `"map"` used by fake_camera_node). Full pipeline integration requires localization to transform camera→map; that's a separate phase.

## Config Additions

In `common/config.py`:

```python
DETECTION_TARGET_CLASS = "cat"
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_MODEL = "yolo11x-seg.pt"  # x-size for accuracy; RTX 4060 handles it
DEPTH_SCALE = 1000.0  # Orbbec publishes depth in millimeters
```

## Dependencies

Added to `requirements.txt`:

```
ultralytics>=8.0
opencv-python>=4.8
pyorbbecsdk>=1.0
```

## How to Run

```bash
# Step 1: Test camera capture (saves sample RGB + depth images)
conda activate lekiwi
python -m perception.test_camera

# Step 2: Test YOLO detection on a cup (saves annotated image with mask overlay)
python -m perception.test_detection --target cup

# Step 3: Run full detection node (publishes on ZMQ)
python -m nodes.real_camera_node
```

Each test script produces visual output (saved images with overlays) for verification.

# Run the full demo with fake robot pose + real detection
# (state_node won't advance past WAIT_FOR_ROBOT without robot_pose,
#  but you can observe detection messages with a ZMQ subscriber)
```

## Limitations / Future Work

- **Camera frame only**: No transform to world/map frame until localization is added.
- **Single target**: Detects highest-confidence instance of the configured class. Multi-object tracking is out of scope.
- **No robot_pose or occupancy_grid**: These topics are not published by this node yet.
- **No GPU requirement**: yolov8n-seg runs on CPU (slower) or GPU (faster). No explicit CUDA dependency; ultralytics handles device selection automatically.
