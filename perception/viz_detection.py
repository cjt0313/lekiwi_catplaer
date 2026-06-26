"""Real-time 3D detection visualization using viser.

Runs camera + YOLO detection, publishes on ZMQ, and serves a viser 3D viewer
showing: full scene point cloud, masked object points, and 3D AABB wireframe.

Usage:
    python -m perception.viz_detection [--target cup] [--fps 3]
    # Open http://localhost:8080 in browser
"""

import sys
import time
import json
import argparse

import cv2
import numpy as np
import zmq
import viser

sys.path.insert(0, ".")

from common.config import (
    FAKE_CAMERA_PUB, DETECTION_TARGET_CLASS,
    DETECTION_CONF_THRESHOLD, DETECTION_MODEL, DEPTH_SCALE,
    APRILTAG_FAMILY, APRILTAG_SIZE, T_BASE_TAG,
    GRID_SIZE, GRID_RESOLUTION, GRID_ORIGIN_OFFSET,
    BASE_HEIGHT_THRESHOLD, ROBOT_RADIUS_CELLS, TARGET_RADIUS_CELLS,
    TARGET_GRID_INFLATE, ROBOT_COLLISION_RADIUS, DESIRED_CAT_DISTANCE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, publish, ZmqMessage
from perception.camera import OrbbecCamera
from perception.detection import (
    load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb,
)
from perception.localization import (
    load_detector, detect_tag_pose, invert_transform, transform_points,
)

# Control signal for robot bridge
CONTROL_SIGNAL_ADDR = "tcp://127.0.0.1:5570"
CONTROL_SIGNAL_TOPIC = "bridge_control"



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



def backproject_full_colored(rgb, depth, K, stride=4, max_depth_mm=3000):
    """Back-project full depth to Nx3 points + Nx3 uint8 colors."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ys_flat = ys.ravel()
    xs_flat = xs.ravel()

    zs = depth[ys_flat, xs_flat].astype(np.float32)
    valid = (zs > 0) & (zs < max_depth_mm)
    xs_v = xs_flat[valid].astype(np.float32)
    ys_v = ys_flat[valid].astype(np.float32)
    zs_v = zs[valid] / DEPTH_SCALE

    X = (xs_v - cx) * zs_v / fx
    Y = (ys_v - cy) * zs_v / fy
    Z = zs_v

    points = np.stack([X, Y, Z], axis=1)
    colors = rgb[ys_flat[valid].astype(int), xs_v.astype(int)]

    return points, colors


def bbox_wireframe(bbox_min, bbox_max):
    """Generate 12-edge wireframe for an AABB. Returns (12, 2, 3) for viser add_line_segments."""
    x0, y0, z0 = bbox_min
    x1, y1, z1 = bbox_max

    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return np.array([[corners[a], corners[b]] for a, b in edges])  # (12, 2, 3)


_GRID_COLORMAP = np.array([
    [128, 128, 128],  # 0: unknown — gray
    [255, 255, 255],  # 1: freespace — white
    [180, 0, 0],      # 2: occupied — dark red
    [0, 100, 255],    # 3: robot — blue
    [0, 255, 100],    # 4: target — green
], dtype=np.uint8)

_ROBOT_CENTER = GRID_SIZE // 2
_YY, _XX = np.ogrid[:GRID_SIZE, :GRID_SIZE]
_ROBOT_MASK = (_XX - _ROBOT_CENTER)**2 + (_YY - _ROBOT_CENTER)**2 <= ROBOT_RADIUS_CELLS**2
_FREESPACE_RADIUS_CELLS = int(0.20 / GRID_RESOLUTION)
_FREESPACE_MASK = (_XX - _ROBOT_CENTER)**2 + (_YY - _ROBOT_CENTER)**2 <= _FREESPACE_RADIUS_CELLS**2
_ASSUMED_FREE_RADIUS_CELLS = int(2.0 / GRID_RESOLUTION)
_ASSUMED_FREE_MASK = (_XX - _ROBOT_CENTER)**2 + (_YY - _ROBOT_CENTER)**2 <= _ASSUMED_FREE_RADIUS_CELLS**2
_CLOSE_KERNEL = np.ones((3, 3), dtype=np.uint8)

_GRID_SCALE = 4


def generate_grid_map(points, center_xy=None, target_radius_m=None):
    """Generate 2D occupancy grid from base-frame points.
    Returns 100x100 uint8: 0=unknown, 1=freespace, 2=occupied, 3=robot, 4=target.
    """
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

    if len(points) > 0:
        cx = ((points[:, 0] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION).astype(int)
        cy = ((points[:, 1] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION).astype(int)
        valid = (cx >= 0) & (cx < GRID_SIZE) & (cy >= 0) & (cy < GRID_SIZE)
        cx, cy, zs = cx[valid], cy[valid], points[valid, 2]

        freespace = zs < BASE_HEIGHT_THRESHOLD
        occupied = ~freespace

        np.maximum.at(grid, (cy[occupied], cx[occupied]), 2)
        free_mask = grid[cy[freespace], cx[freespace]] == 0
        grid[cy[freespace][free_mask], cx[freespace][free_mask]] = 1

    # Force 20cm radius around robot as freespace (overrides occupied from robot body)
    grid[_FREESPACE_MASK] = 1

    # Morphological close to fill small holes in freespace
    free_binary = (grid == 1).astype(np.uint8)
    closed = cv2.morphologyEx(free_binary, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    grid[(closed == 1) & (grid == 0)] = 1

    # Assume unknown cells within 2m of robot are freespace
    grid[_ASSUMED_FREE_MASK & (grid == 0)] = 1

    grid[_ROBOT_MASK] = 3

    if center_xy is not None:
        tx = int((center_xy[0] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION)
        ty = int((center_xy[1] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION)
        if 0 <= tx < GRID_SIZE and 0 <= ty < GRID_SIZE:
            if target_radius_m is not None:
                r_cells = int(target_radius_m * TARGET_GRID_INFLATE / GRID_RESOLUTION)
            else:
                r_cells = TARGET_RADIUS_CELLS
            target_mask = (_XX - tx)**2 + (_YY - ty)**2 <= r_cells**2
            grid[target_mask] = 4

    return grid


_INFLATE_CELLS = int(ROBOT_COLLISION_RADIUS / GRID_RESOLUTION)
_INFLATE_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * _INFLATE_CELLS + 1, 2 * _INFLATE_CELLS + 1)
)


def inflate_obstacles(grid):
    """Inflate occupied cells by robot collision radius. Returns binary obstacle map."""
    obstacle = (grid == 2).astype(np.uint8)
    inflated = cv2.dilate(obstacle, _INFLATE_KERNEL)
    # Robot area is always traversable (we're already there)
    inflated[grid == 3] = 0
    return inflated


def find_free_goal(inflated, goal_rc, max_radius=15):
    """Find nearest free cell to goal_rc in inflated map. Spiral search."""
    gr, gc = goal_rc
    if 0 <= gr < GRID_SIZE and 0 <= gc < GRID_SIZE and inflated[gr, gc] == 0:
        return goal_rc
    for r in range(1, max_radius + 1):
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                if abs(dr) != r and abs(dc) != r:
                    continue
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and inflated[nr, nc] == 0:
                    return (nr, nc)
    return None


def astar_grid(grid, start_rc, goal_rc):
    """A* on 100x100 grid. Returns path as list of (row, col) or empty list."""
    import heapq

    sr, sc = start_rc
    gr, gc = goal_rc

    if not (0 <= gr < GRID_SIZE and 0 <= gc < GRID_SIZE):
        return []

    inflated = inflate_obstacles(grid)

    # Find collision-free goal near requested goal
    valid_goal = find_free_goal(inflated, goal_rc)
    if valid_goal is None:
        return []
    gr, gc = valid_goal

    # Start is always free (robot is there)
    inflated[sr, sc] = 0

    def h(r, c):
        return ((r - gr)**2 + (c - gc)**2) ** 0.5

    open_set = [(h(sr, sc), 0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    closed = set()

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, cost, r, c = heapq.heappop(open_set)
        if (r, c) in closed:
            continue
        closed.add((r, c))

        if r == gr and c == gc:
            path = []
            cur = (gr, gc)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append((sr, sc))
            path.reverse()
            return path

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
                continue
            if inflated[nr, nc] == 1:
                continue
            if (nr, nc) in closed:
                continue

            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            new_g = g_score[(r, c)] + move_cost
            if new_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = new_g
                heapq.heappush(open_set, (new_g + h(nr, nc), new_g, nr, nc))
                came_from[(nr, nc)] = (r, c)

    return []


def draw_path_on_grid(grid_img, path, scale=_GRID_SCALE, color=(0, 0, 0)):
    """Draw path as yellow line on the upscaled grid image."""
    if len(path) < 2:
        return
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        pt0 = (c0 * scale + scale // 2, r0 * scale + scale // 2)
        pt1 = (c1 * scale + scale // 2, r1 * scale + scale // 2)
        cv2.line(grid_img, pt0, pt1, color, 2)


def path_to_world(path):
    """Convert grid path [(row, col), ...] to base-frame [(x, y), ...]."""
    world_path = []
    for r, c in path:
        x = c * GRID_RESOLUTION - GRID_ORIGIN_OFFSET
        y = r * GRID_RESOLUTION - GRID_ORIGIN_OFFSET
        world_path.append((x, y))
    return world_path


def main():
    parser = argparse.ArgumentParser(description="Real-time 3D detection visualizer")
    parser.add_argument("--target", default=DETECTION_TARGET_CLASS)
    parser.add_argument("--fps", type=float, default=3.0, help="Visualization update rate")
    parser.add_argument("--stride", type=int, default=4, help="Point cloud downsample stride")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--no-zmq", action="store_true", help="Disable ZMQ publishing")
    parser.add_argument("--localize", action="store_true",
                        help="Enable AprilTag-based frame transform (tag = world origin)")
    parser.add_argument("--distance", type=float, default=DESIRED_CAT_DISTANCE,
                        help="Desired standoff distance from target (meters)")
    args = parser.parse_args()

    # Initialize
    print(f"Loading YOLO model: {DETECTION_MODEL}")
    model = load_model(DETECTION_MODEL)

    print("Starting camera...")
    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    pub = None if args.no_zmq else make_publisher(FAKE_CAMERA_PUB)

    # AprilTag localization
    tag_detector = None
    if args.localize:
        print(f"Loading AprilTag detector: {APRILTAG_FAMILY}, size={APRILTAG_SIZE}m")
        tag_detector = load_detector(APRILTAG_FAMILY)

    # Start viser server
    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/camera", axes_length=0.1, axes_radius=0.003)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    if args.localize:
        server.scene.add_frame("/tag_origin", axes_length=0.1, axes_radius=0.004)

    # GUI elements
    gui_fps = server.gui.add_number("FPS", initial_value=0, disabled=True)
    gui_status = server.gui.add_text("Status", initial_value="Starting...", disabled=True)
    gui_target = server.gui.add_text("Target", initial_value=args.target, disabled=True)

    # Robot control toggle button
    control_pub = zmq.Context.instance().socket(zmq.PUB)
    control_pub.bind(CONTROL_SIGNAL_ADDR)
    robot_enabled = False
    gui_robot_btn = server.gui.add_button("Robot: OFF", color="red")

    @gui_robot_btn.on_click
    def _toggle_robot(event):
        nonlocal robot_enabled
        robot_enabled = not robot_enabled
        if robot_enabled:
            gui_robot_btn.name = "Robot: ON"
            gui_robot_btn.color = "green"
        else:
            gui_robot_btn.name = "Robot: OFF"
            gui_robot_btn.color = "red"
        msg = json.dumps({"enabled": robot_enabled}).encode("utf-8")
        control_pub.send_multipart([CONTROL_SIGNAL_TOPIC.encode("utf-8"), msg])

    gui_estop_btn = server.gui.add_button("E-STOP", color="red")

    @gui_estop_btn.on_click
    def _estop(event):
        nonlocal robot_enabled
        robot_enabled = False
        gui_robot_btn.name = "Robot: OFF"
        gui_robot_btn.color = "red"
        msg = json.dumps({"enabled": False, "estop": True}).encode("utf-8")
        control_pub.send_multipart([CONTROL_SIGNAL_TOPIC.encode("utf-8"), msg])

    gui_rgb = server.gui.add_image(
        np.zeros((480, 640, 3), dtype=np.uint8),
        label="Camera",
    )
    gui_grid = server.gui.add_image(
        np.zeros((GRID_SIZE * _GRID_SCALE, GRID_SIZE * _GRID_SCALE, 3), dtype=np.uint8),
        label="Grid Map",
    )

    print(f"\nVisualization running at http://localhost:{args.port}")
    print(f"Target: '{args.target}', update rate: {args.fps} Hz, stride: {args.stride}")
    print("Press Ctrl+C to stop.\n")

    interval = 1.0 / args.fps
    seq = 0
    frame_count = 0
    fps_timer = time.time()

    try:
        while True:
            t_start = time.time()

            rgb, depth = camera.grab()
            if rgb is None:
                continue

            gui_rgb.image = rgb

            # Detection
            result = detect_target(model, rgb, args.target, DETECTION_CONF_THRESHOLD)

            # Full scene point cloud (RGB-colored)
            points_full, colors_full = backproject_full_colored(
                rgb, depth, K, stride=args.stride)

            if result is not None:
                mask, confidence = result
                obj_points = mask_to_3d_points(mask, depth, K, depth_scale=DEPTH_SCALE)
                obj_points = remove_outliers(obj_points)

                if len(obj_points) >= 10:
                    center, bbox_min, bbox_max = fit_aabb(obj_points)
                    size = bbox_max - bbox_min
                    detected = True
                else:
                    detected = False
            else:
                detected = False

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

            # Update viser scene
            server.scene.add_point_cloud(
                "/scene/points",
                points=points_full.astype(np.float32),
                colors=colors_full,
                point_size=0.003,
            )

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

            # 2D Grid map (rendered on GUI panel)
            if T_base_camera is not None:
                target_xy = center[:2] if detected else None
                target_r = max(size[0], size[1]) / 2 if detected else None
                grid = generate_grid_map(points_full, center_xy=target_xy, target_radius_m=target_r)
                grid_img = _GRID_COLORMAP[grid]
                grid_img = np.repeat(np.repeat(grid_img, _GRID_SCALE, axis=0), _GRID_SCALE, axis=1)

                # A* path planning: maintain standoff distance from target
                world_path = None
                if detected:
                    start_rc = (_ROBOT_CENTER, _ROBOT_CENTER)
                    robot_xy = np.array([0.0, 0.0])
                    dir_to_robot = robot_xy - target_xy
                    dist = np.linalg.norm(dir_to_robot)
                    if abs(dist - args.distance) > 0.05 and dist > 0.01:
                        standoff_xy = target_xy + dir_to_robot / dist * args.distance
                        goal_rc = (
                            int((standoff_xy[1] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION),
                            int((standoff_xy[0] + GRID_ORIGIN_OFFSET) / GRID_RESOLUTION),
                        )
                        path = astar_grid(grid, start_rc, goal_rc)
                        if path:
                            draw_path_on_grid(grid_img, path)
                            world_path = path_to_world(path)

                gui_grid.image = grid_img

            if detected:
                # Mask points in green
                server.scene.add_point_cloud(
                    "/scene/mask_points",
                    points=obj_points.astype(np.float32),
                    colors=np.full((len(obj_points), 3), (0, 255, 0), dtype=np.uint8),
                    point_size=0.006,
                )

                # AABB wireframe
                wire = bbox_wireframe(bbox_min, bbox_max)
                server.scene.add_line_segments(
                    "/scene/bbox",
                    points=wire.astype(np.float32),
                    colors=(255, 255, 0),
                    line_width=3.0,
                )

                # Center marker
                server.scene.add_frame(
                    "/scene/center",
                    position=tuple(center),
                    axes_length=0.03,
                    axes_radius=0.002,
                )

                gui_status.value = (
                    f"{gui_status_prefix}DETECTED conf={confidence:.2f} "
                    f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) "
                    f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})"
                )

                # ZMQ publish
                if pub is not None:
                    payload = {
                        "center": center.tolist(),
                        "bbox_min": bbox_min.tolist(),
                        "bbox_max": bbox_max.tolist(),
                        "confidence": confidence,
                        "visible": True,
                        "path": world_path,
                    }
                    msg = ZmqMessage.create(MsgType.CAT_BBOX_3D, payload,
                                            "viz_detection", seq=seq, frame_id="camera")
                    publish(pub, msg)
            else:
                # Clear mask and bbox when no detection
                server.scene.add_point_cloud(
                    "/scene/mask_points",
                    points=np.zeros((1, 3), dtype=np.float32),
                    colors=np.zeros((1, 3), dtype=np.uint8),
                    point_size=0.001,
                )
                server.scene.add_line_segments(
                    "/scene/bbox",
                    points=np.zeros((1, 2, 3), dtype=np.float32),
                    colors=(0, 0, 0),
                    line_width=1.0,
                )
                gui_status.value = f"{gui_status_prefix}No detection"

                if pub is not None:
                    payload = {
                        "center": None, "bbox_min": None, "bbox_max": None,
                        "confidence": 0.0, "visible": False,
                    }
                    msg = ZmqMessage.create(MsgType.CAT_BBOX_3D, payload,
                                            "viz_detection", seq=seq, frame_id="camera")
                    publish(pub, msg)

            seq += 1
            frame_count += 1

            # FPS tracking
            now = time.time()
            if now - fps_timer >= 1.0:
                gui_fps.value = round(frame_count / (now - fps_timer), 1)
                fps_timer = now
                frame_count = 0

            # Throttle to target FPS
            elapsed = time.time() - t_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
