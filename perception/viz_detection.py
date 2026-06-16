"""Real-time 3D detection visualization using viser.

Runs camera + YOLO detection, publishes on ZMQ, and serves a viser 3D viewer
showing: full scene point cloud, masked object points, and 3D AABB wireframe.

Usage:
    python -m perception.viz_detection [--target cup] [--fps 3]
    # Open http://localhost:8080 in browser
"""

import sys
import time
import argparse

import numpy as np
import viser

sys.path.insert(0, ".")

from common.config import (
    FAKE_CAMERA_PUB, DETECTION_TARGET_CLASS,
    DETECTION_CONF_THRESHOLD, DETECTION_MODEL, DEPTH_SCALE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, publish, ZmqMessage
from perception.camera import OrbbecCamera
from perception.detection import (
    load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb,
)



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


def main():
    parser = argparse.ArgumentParser(description="Real-time 3D detection visualizer")
    parser.add_argument("--target", default=DETECTION_TARGET_CLASS)
    parser.add_argument("--fps", type=float, default=3.0, help="Visualization update rate")
    parser.add_argument("--stride", type=int, default=4, help="Point cloud downsample stride")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    parser.add_argument("--no-zmq", action="store_true", help="Disable ZMQ publishing")
    args = parser.parse_args()

    # Initialize
    print(f"Loading YOLO model: {DETECTION_MODEL}")
    model = load_model(DETECTION_MODEL)

    print("Starting camera...")
    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    pub = None if args.no_zmq else make_publisher(FAKE_CAMERA_PUB)

    # Start viser server
    server = viser.ViserServer(port=args.port)
    server.scene.add_frame("/camera", axes_length=0.1, axes_radius=0.003)
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)

    # GUI elements
    gui_fps = server.gui.add_number("FPS", initial_value=0, disabled=True)
    gui_status = server.gui.add_text("Status", initial_value="Starting...", disabled=True)
    gui_target = server.gui.add_text("Target", initial_value=args.target, disabled=True)

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

            # Update viser scene
            server.scene.add_point_cloud(
                "/scene/points",
                points=points_full.astype(np.float32),
                colors=colors_full,
                point_size=0.003,
            )

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
                    f"DETECTED conf={confidence:.2f} "
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
                gui_status.value = "No detection"

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
