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
