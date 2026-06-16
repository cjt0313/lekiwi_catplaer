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

    for _ in range(20):
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
