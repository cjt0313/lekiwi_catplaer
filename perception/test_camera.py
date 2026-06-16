"""Test Orbbec camera capture. Saves sample RGB + depth images for visual verification."""

import sys
import cv2
import numpy as np

sys.path.insert(0, ".")
from perception.camera import OrbbecCamera


def colorize_depth(depth, max_depth_mm=5000):
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

    if rgb is not None and depth is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_vis = colorize_depth(depth)

        cv2.imwrite("test_output_rgb.png", bgr)
        cv2.imwrite("test_output_depth.png", depth_vis)

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
