# AprilTag Base Frame Transform + 2D Grid Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform point cloud to robot base frame via AprilTag + URDF FK, then project into a 2D occupancy grid visualized in viser.

**Architecture:** AprilTag detected on Top-V2-v2 link gives T_camera_tag. A static T_base_tag (from URDF FK via yourdfpy) chains to get T_base_camera. Point cloud transforms to base frame, then projects to 2D grid (100x100, 2.5cm cells). Grid rendered as flat image in viser.

**Tech Stack:** yourdfpy (URDF parsing), numpy, viser, pupil_apriltags

---

### Task 1: Create URDF with AprilTag Link

**Files:**
- Create: `/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf`

- [ ] **Step 1: Copy original URDF and add apriltag_link**

Copy `/home/jim/code/VLA/LeKiwi/URDF/LeKiwi.urdf` to `/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf`, then add before the closing `</robot>` tag (after all existing joints):

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

The Z offset (0.027m) places the tag on top of the Top-V2-v2 surface. The `rpy` in the joint origin is what you rotate to calibrate:
- Z (yaw): spin tag in-plane
- X (roll): tilt left/right
- Y (pitch): tilt forward/back

- [ ] **Step 2: Verify URDF is valid**

Run:
```bash
conda run -n lekiwi pip install yourdfpy && conda run -n lekiwi python -c "
import yourdfpy
urdf = yourdfpy.URDF.load('/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf', build_scene_graph=True, load_meshes=False)
print('Links:', [l for l in urdf.link_map.keys() if 'april' in l.lower()])
print('Joints:', [j for j in urdf.joint_map.keys() if 'april' in j.lower()])
"
```

Expected: `Links: ['apriltag_link']` and `Joints: ['Top-V2-v2_to_apriltag']`

---

### Task 2: Compute and Save T_base_tag

**Files:**
- Create: `scripts/compute_T_base_tag.py`
- Create: `common/T_base_tag.npy`

- [ ] **Step 1: Create scripts directory and calibration script**

Create `scripts/compute_T_base_tag.py`:

```python
"""Compute static T_base_tag from URDF and save as .npy."""

import sys
from pathlib import Path
import numpy as np
import yourdfpy

URDF_PATH = Path("/home/jim/code/VLA/LeKiwi/URDF/LeKiwi_apriltag.urdf")
OUTPUT_PATH = Path(__file__).parent.parent / "common" / "T_base_tag.npy"

def main():
    urdf = yourdfpy.URDF.load(
        str(URDF_PATH),
        build_scene_graph=True,
        load_meshes=False,
        load_collision_meshes=False,
    )
    T_base_tag = urdf.get_transform("apriltag_link", frame_from="base_plate_layer1-v5")
    print("T_base_tag:")
    print(T_base_tag)
    print(f"\nTranslation: {T_base_tag[:3, 3]}")
    np.save(str(OUTPUT_PATH), T_base_tag)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run calibration script**

Run:
```bash
conda run -n lekiwi python scripts/compute_T_base_tag.py
```

Expected: prints a 4x4 matrix with a translation Z component around 0.05-0.1m (height of tag above base plate through the kinematic chain). File `common/T_base_tag.npy` created.

- [ ] **Step 3: Verify the saved matrix**

Run:
```bash
conda run -n lekiwi python -c "import numpy as np; T = np.load('common/T_base_tag.npy'); print(T); print('det(R):', np.linalg.det(T[:3,:3]))"
```

Expected: det(R) ≈ 1.0 (valid rotation), translation Z > 0 (tag is above base).

---

### Task 3: Update config.py with T_BASE_TAG and Grid Parameters

**Files:**
- Modify: `common/config.py`

- [ ] **Step 1: Add T_BASE_TAG loading and grid map parameters**

Add to the end of `common/config.py`:

```python
# Base-frame transform (from URDF FK, computed by scripts/compute_T_base_tag.py)
import os as _os
_T_BASE_TAG_PATH = _os.path.join(_os.path.dirname(__file__), "T_base_tag.npy")
T_BASE_TAG = np.load(_T_BASE_TAG_PATH) if _os.path.exists(_T_BASE_TAG_PATH) else None

# Perception grid map parameters (2D occupancy)
GRID_RESOLUTION = 0.025  # meters per cell
GRID_SIZE = 100  # cells (100x100)
GRID_PHYSICAL_SIZE = GRID_SIZE * GRID_RESOLUTION  # 2.5m
GRID_ORIGIN_OFFSET = GRID_PHYSICAL_SIZE / 2  # 1.25m — grid center = robot base
BASE_HEIGHT_THRESHOLD = 0.01  # meters — points below this Z are floor/freespace
ROBOT_RADIUS_CELLS = int(0.15 / GRID_RESOLUTION)  # ~6 cells
TARGET_RADIUS_CELLS = int(0.05 / GRID_RESOLUTION)  # ~2 cells default
```

Also add `import numpy as np` at the top of config.py if not already present.

---

### Task 4: Update viz_detection.py — Base Frame Transform

**Files:**
- Modify: `perception/viz_detection.py`

- [ ] **Step 1: Update imports**

Add `T_BASE_TAG` to the config imports (line 20-24):

```python
from common.config import (
    FAKE_CAMERA_PUB, DETECTION_TARGET_CLASS,
    DETECTION_CONF_THRESHOLD, DETECTION_MODEL, DEPTH_SCALE,
    APRILTAG_FAMILY, APRILTAG_SIZE, T_BASE_TAG,
    GRID_SIZE, GRID_RESOLUTION, GRID_ORIGIN_OFFSET,
    BASE_HEIGHT_THRESHOLD, ROBOT_RADIUS_CELLS, TARGET_RADIUS_CELLS,
)
```

- [ ] **Step 2: Replace tag-frame transform with base-frame transform**

Replace the frame transform block (lines 191-208) with:

```python
            # Frame transform (localization → base frame)
            gui_status_prefix = ""
            T_base_camera = None
            if tag_detector is not None and T_BASE_TAG is not None:
                T_camera_tag = detect_tag_pose(tag_detector, rgb, K, tag_size=APRILTAG_SIZE)
                if T_camera_tag is not None:
                    T_tag_camera = invert_transform(T_camera_tag)
                    T_base_camera = T_BASE_TAG @ T_tag_camera
                    points_full = transform_points(points_full, T_base_camera)
                    if detected:
                        obj_points = transform_points(obj_points, T_base_camera)
                        center, bbox_min, bbox_max = fit_aabb(obj_points)
                        size = bbox_max - bbox_min
                    gui_status_prefix = "[BASE] "
                else:
                    gui_status_prefix = "[TAG LOST] "
```

- [ ] **Step 3: Update camera pose visualization**

Replace the camera pose section (lines 218-228) to use `T_base_camera`:

```python
            # Camera pose in base frame
            if T_base_camera is not None:
                cam_pos = T_base_camera[:3, 3]
                cam_quat = rotation_matrix_to_quaternion(T_base_camera[:3, :3])
                server.scene.add_frame(
                    "/scene/camera_pose",
                    position=tuple(cam_pos),
                    wxyz=cam_quat,
                    axes_length=0.05,
                    axes_radius=0.002,
                )
```

---

### Task 5: Add 2D Grid Map Generation and Visualization

**Files:**
- Modify: `perception/viz_detection.py`

- [ ] **Step 1: Add grid map generation function**

Add after the `bbox_wireframe` function (after line 111):

```python
def generate_grid_map(points, center_xy=None, target_radius_cells=TARGET_RADIUS_CELLS):
    """Generate 2D occupancy grid from base-frame points.
    
    Returns 100x100 uint8 array:
      0=unknown, 1=freespace, 2=occupied, 3=robot, 4=target
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    if len(points) > 0:
        # Map XY to grid cells
        cx = ((points[:, 0] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION).astype(int)
        cy = ((points[:, 1] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION).astype(int)
        valid = (cx >= 0) & (cx < GRID_SIZE) & (cy >= 0) & (cy < GRID_SIZE)
        cx, cy, zs = cx[valid], cy[valid], points[valid, 2]

        # For each cell, check if points are below threshold (freespace) or above (occupied)
        for i in range(len(cx)):
            cell_x, cell_y, z = cx[i], cy[i], zs[i]
            if z >= BASE_HEIGHT_THRESHOLD:
                grid[cell_y, cell_x] = 2  # occupied wins
            elif grid[cell_y, cell_x] == 0:
                grid[cell_y, cell_x] = 1  # freespace if not already occupied

    # Robot circle at center
    robot_center = GRID_SIZE // 2
    yy, xx = np.ogrid[:GRID_SIZE, :GRID_SIZE]
    robot_mask = (xx - robot_center)**2 + (yy - robot_center)**2 <= ROBOT_RADIUS_CELLS**2
    grid[robot_mask] = 3

    # Target circle
    if center_xy is not None:
        tx = int((center_xy[0] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION)
        ty = int((center_xy[1] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION)
        if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
            target_mask = (xx - tx)**2 + (yy - ty)**2 <= target_radius_cells**2
            grid[target_mask] = 4

    return grid


def grid_to_rgb(grid):
    """Convert grid map to RGB image for visualization."""
    colormap = np.array([
        [128, 128, 128],  # 0: unknown — gray
        [255, 255, 255],  # 1: freespace — white
        [180, 0, 0],      # 2: occupied — dark red
        [0, 100, 255],    # 3: robot — blue
        [0, 255, 100],    # 4: target — green
    ], dtype=np.uint8)
    return colormap[grid]
```

- [ ] **Step 2: Add grid visualization in the main loop**

After the point cloud is added to viser (after line ~216 `server.scene.add_point_cloud("/scene/points", ...)`), add:

```python
            # 2D Grid map
            if T_base_camera is not None:
                target_xy = center[:2] if detected else None
                grid = generate_grid_map(points_full, center_xy=target_xy)
                grid_rgb = grid_to_rgb(grid)
                # Render as flat image at Z=0
                server.scene.add_mesh_simple(
                    "/scene/grid_map",
                    vertices=np.array([
                        [-GRID_ORIGIN_OFFSET, -GRID_ORIGIN_OFFSET, 0.0],
                        [GRID_ORIGIN_OFFSET, -GRID_ORIGIN_OFFSET, 0.0],
                        [GRID_ORIGIN_OFFSET, GRID_ORIGIN_OFFSET, 0.0],
                        [-GRID_ORIGIN_OFFSET, GRID_ORIGIN_OFFSET, 0.0],
                    ], dtype=np.float32),
                    faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
                    color=(200, 200, 200),
                    flat_shading=True,
                    opacity=0.7,
                )
```

Note: viser's `add_mesh_simple` doesn't support per-face textures directly. Alternative — use `add_point_cloud` with a flat grid of colored points at Z=0:

```python
            # 2D Grid map (as colored point cloud at Z=0)
            if T_base_camera is not None:
                target_xy = center[:2] if detected else None
                grid = generate_grid_map(points_full, center_xy=target_xy)
                grid_rgb = grid_to_rgb(grid)
                # Create grid point positions
                gx = np.linspace(-GRID_ORIGIN_OFFSET + GRID_RESOLUTION/2,
                                 GRID_ORIGIN_OFFSET - GRID_RESOLUTION/2, GRID_SIZE)
                gy = np.linspace(-GRID_ORIGIN_OFFSET + GRID_RESOLUTION/2,
                                 GRID_ORIGIN_OFFSET - GRID_RESOLUTION/2, GRID_SIZE)
                gxx, gyy = np.meshgrid(gx, gy)
                grid_points = np.stack([
                    gxx.ravel(), gyy.ravel(), np.zeros(GRID_SIZE * GRID_SIZE)
                ], axis=1).astype(np.float32)
                grid_colors = grid_rgb.reshape(-1, 3)
                server.scene.add_point_cloud(
                    "/scene/grid_map",
                    points=grid_points,
                    colors=grid_colors,
                    point_size=GRID_RESOLUTION,
                )
```

Use the point cloud approach — it's simpler and works reliably with viser.

- [ ] **Step 3: Run and verify**

Run:
```bash
conda run -n lekiwi python -m perception.viz_detection --localize --target cup
```

Open http://localhost:8080. Verify:
- Point cloud is in base frame (floor at Z ≈ -base_height)
- Grid map visible as colored flat layer at Z=0
- Robot circle (blue) at center
- Target circle (green) at detected object position
- Freespace (white) where floor is visible
- Occupied (red) where obstacles are

- [ ] **Step 4: Commit all changes**

```bash
git add -A
git commit -m "feat: transform point cloud to base frame + 2D grid map visualization"
```

---

### Task 6: Final Integration Test

- [ ] **Step 1: Run full pipeline check**

```bash
conda run -n lekiwi python -m perception.viz_detection --localize --target cup --fps 3
```

Verify in browser:
1. `[BASE]` prefix in status bar (tag detected, base-frame mode active)
2. 3D point cloud correctly oriented (floor flat, walls vertical)
3. Grid map overlay at Z=0 with correct classifications
4. Target green circle tracks the detected object
5. No Z-flip issues

- [ ] **Step 2: Test tag-lost fallback**

Cover the AprilTag — verify:
- Status shows `[TAG LOST]`
- Point cloud stays in camera frame (no transform applied)
- Grid map disappears (only shown when tag is detected)
