# AprilTag Base Frame Transform + 2D Grid Map

## Summary

Add AprilTag link to the LeKiwi URDF, transform point cloud from camera frame all the way to robot base frame (using the static FK chain), and project into a 2D occupancy grid map visualized in the existing viser scene.

## 1. URDF: AprilTag Link

**File**: `/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf` (new file, copy of LeKiwi.urdf with additions)

Add after `Top-V2-v2` link definition:

```xml
<link name="apriltag_link">
    <visual name="apriltag_visual">
        <origin xyz="0 0 0" rpy="0 0 0" />
        <geometry>
            <box size="0.023 0.023 0.001" />
        </geometry>
    </visual>
</link>

<joint name="Top-V2-v2_to_apriltag" type="fixed">
    <origin xyz="0.0 0.0 0.027" rpy="0 0 0" />
    <parent link="Top-V2-v2" />
    <child link="apriltag_link" />
</joint>
```

**Calibration axis guidance**:
- **Z rotation (yaw)**: Spin tag in-plane on the top surface. This is the primary DOF for aligning the tag's X/Y axes with the robot's forward direction.
- **X rotation (roll)**: Tilt tag left/right relative to the surface.
- **Y rotation (pitch)**: Tilt tag forward/backward.

Adjust the `rpy` values in the joint origin to calibrate the tag's orientation. The Z offset (0.027m) positions the tag on top of the Top-V2-v2 surface — adjust if physical mounting differs.

## 2. Compute T_base_tag (Calibration Script)

**File**: `scripts/compute_T_base_tag.py`

**Method** (same as eye_to_hand_usage repo):
```python
import yourdfpy
import numpy as np

urdf = yourdfpy.URDF.load("path/to/LeKiwi_apriltag.urdf", ...)
T_base_tag = urdf.get_transform("apriltag_link", frame_from="base_plate_layer1-v5")
np.save("common/T_base_tag.npy", T_base_tag)
```

**Output**: `common/T_base_tag.npy` — a 4x4 float64 matrix.

**Config** (`common/config.py`):
```python
import os
T_BASE_TAG = np.load(os.path.join(os.path.dirname(__file__), "T_base_tag.npy"))
```

## 3. Transform Pipeline (in viz_detection.py)

Replace current tag-frame transform with base-frame transform:

```
T_camera_tag = detect_tag_pose(...)          # from pupil_apriltags
T_tag_camera = invert(T_camera_tag)          # invert
T_base_camera = T_BASE_TAG @ T_tag_camera    # chain to base frame
points_base = transform_points(points_camera, T_base_camera)
```

No Z-flip correction needed — the URDF FK chain naturally encodes the correct orientation from base to tag (including whatever rotation the tag has relative to the base).

## 4. 2D Grid Map

### Parameters
- Grid size: 100x100 cells
- Cell size: 0.025m (2.5cm)
- Physical area: 2.5m x 2.5m
- Origin: robot base center = grid center (cell 50, 50)

### Cell Classification

For each frame:
1. Project all base-frame points to grid cells via: `cell_x = int((point_x + 1.25) / 0.025)`, `cell_y = int((point_y + 1.25) / 0.025)`
2. For each occupied cell:
   - If ALL points in that cell have `Z < BASE_HEIGHT` (floor level, ~0.0m since base is the reference) → **freespace**
   - If ANY point in that cell has `Z >= BASE_HEIGHT` → **occupied**
3. Cells with no points → **unknown**

### Overlays
- **Robot**: filled circle at grid center, radius = robot footprint (~0.15m → 6 cells)
- **Target**: filled circle at detected object center (x,y) in base frame, radius from AABB footprint

### Cell Values
- 0 = unknown (gray)
- 1 = freespace (white/light)
- 2 = occupied (dark/red)
- 3 = robot (blue circle)
- 4 = target (green circle)

### Visualization in Viser

Render as a flat textured mesh (or `add_image` on a plane) at Z=0 in the base frame scene. The 100x100 grid becomes an RGB image:
- Unknown: gray (128, 128, 128)
- Freespace: white (255, 255, 255)
- Occupied: dark red (180, 0, 0)
- Robot: blue (0, 100, 255)
- Target: green (0, 255, 100)

Add as `/scene/grid_map` in viser alongside the existing point cloud.

## 5. Files Changed/Created

| File | Action |
|------|--------|
| `/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf` | New — URDF with apriltag_link |
| `scripts/compute_T_base_tag.py` | New — one-shot calibration script |
| `common/T_base_tag.npy` | New — static transform matrix |
| `common/config.py` | Edit — add T_BASE_TAG, grid params |
| `perception/viz_detection.py` | Edit — base-frame transform + grid map rendering |

## 6. Dependencies

- `yourdfpy` (pip install, used only by calibration script)

## 7. Verification

Manual visual checks (no automated test suite):
1. Run calibration script, print T_base_tag, confirm translation matches expected offset from URDF chain (~0.05-0.1m Z offset from base)
2. Run `python -m perception.viz_detection --localize --target cup`:
   - Floor appears at Z ≈ -BASE_HEIGHT (below base frame origin)
   - Point cloud orientation matches physical reality (robot forward = +X or expected axis)
   - Grid map shows floor as freespace, walls/furniture as occupied
   - Robot circle at center, target circle at detected object position
