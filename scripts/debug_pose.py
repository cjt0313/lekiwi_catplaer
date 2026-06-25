"""Debug script: capture one frame, apply transform, save diagnostic images."""

import sys
import numpy as np
import cv2

sys.path.insert(0, ".")

from common.config import (
    DEPTH_SCALE, APRILTAG_FAMILY, APRILTAG_SIZE, T_BASE_TAG,
)
from perception.camera import OrbbecCamera
from perception.localization import load_detector, detect_tag_pose, invert_transform, transform_points


def backproject(depth, K, stride=2, max_depth_mm=3000):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ys_flat, xs_flat = ys.ravel(), xs.ravel()
    zs = depth[ys_flat, xs_flat].astype(np.float32)
    valid = (zs > 0) & (zs < max_depth_mm)
    xs_v = xs_flat[valid].astype(np.float32)
    ys_v = ys_flat[valid].astype(np.float32)
    zs_v = zs[valid] / DEPTH_SCALE
    X = (xs_v - cx) * zs_v / fx
    Y = (ys_v - cy) * zs_v / fy
    Z = zs_v
    return np.stack([X, Y, Z], axis=1)


def save_topdown(points, path, size=500, meters=2.5):
    """Save top-down XY view of points. X=right, Y=up in image."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    img[:] = 40

    scale = size / meters
    cx, cy = size // 2, size // 2

    for m in np.arange(-1.0, 1.5, 0.5):
        px = int(cx + m * scale)
        py = int(cy - m * scale)
        cv2.line(img, (px, 0), (px, size), (60, 60, 60), 1)
        cv2.line(img, (0, py), (size, py), (60, 60, 60), 1)

    cv2.arrowedLine(img, (cx, cy), (cx + 50, cy), (0, 0, 255), 2, tipLength=0.3)
    cv2.arrowedLine(img, (cx, cy), (cx, cy - 50), (0, 255, 0), 2, tipLength=0.3)
    cv2.putText(img, "X", (cx + 55, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(img, "Y", (cx + 5, cy - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    if len(points) > 0:
        px = (cx + points[:, 0] * scale).astype(int)
        py = (cy - points[:, 1] * scale).astype(int)
        valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
        px, py = px[valid], py[valid]
        z_vals = points[valid, 2]
        z_norm = np.clip((z_vals + 0.1) / 0.5, 0, 1)
        for i in range(len(px)):
            c = int(z_norm[i] * 255)
            img[py[i], px[i]] = (255 - c, 0, c)

    cv2.circle(img, (cx, cy), int(0.15 * scale), (255, 200, 0), 2)
    cv2.putText(img, "ROBOT", (cx - 25, cy + int(0.15 * scale) + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

    cv2.imwrite(path, img)
    print(f"Saved top-down view: {path}")


def main():
    print("Starting camera...")
    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    detector = load_detector(APRILTAG_FAMILY)

    print("Capturing frames (waiting for tag)...")
    for attempt in range(30):
        rgb, depth = camera.grab()
        if rgb is None:
            continue
        T_camera_tag = detect_tag_pose(detector, rgb, K, tag_size=APRILTAG_SIZE)
        if T_camera_tag is not None:
            print(f"\nTag detected on attempt {attempt + 1}")
            break
    else:
        print("No tag detected after 30 frames!")
        camera.stop()
        return

    T_tag_camera = invert_transform(T_camera_tag)
    T_base_camera = T_BASE_TAG @ T_tag_camera

    print("\nT_camera_tag:")
    print(np.array2string(T_camera_tag, precision=4))
    print("\nT_base_camera:")
    print(np.array2string(T_base_camera, precision=4))
    print(f"\nCamera position in base frame: {T_base_camera[:3, 3]}")

    points_cam = backproject(depth, K)
    points_base = transform_points(points_cam, T_base_camera)

    print(f"\nPoint cloud stats (base frame):")
    print(f"  X range: [{points_base[:, 0].min():.3f}, {points_base[:, 0].max():.3f}]")
    print(f"  Y range: [{points_base[:, 1].min():.3f}, {points_base[:, 1].max():.3f}]")
    print(f"  Z range: [{points_base[:, 2].min():.3f}, {points_base[:, 2].max():.3f}]")
    print(f"  Median Z: {np.median(points_base[:, 2]):.3f}")
    print(f"  Points below Z=0 (floor): {(points_base[:, 2] < 0).sum()} / {len(points_base)}")

    save_topdown(points_base, "debug_topdown.png")
    cv2.imwrite("debug_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("Saved debug_rgb.png")

    camera.stop()
    print("\nDone. Check debug_topdown.png and debug_rgb.png")


if __name__ == "__main__":
    main()
