"""System configuration: ZMQ addresses and parameters."""

import os as _os
import numpy as np

# ZMQ PUB addresses
FAKE_CAMERA_PUB = "tcp://127.0.0.1:5560"

# Robot parameters
DESIRED_CAT_DISTANCE = 0.2  # meters
ROBOT_COLLISION_RADIUS = 0.20  # meters (inflated from 15cm physical)

# Detection parameters
DETECTION_TARGET_CLASS = "cat"
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_MODEL = "yolo11x-seg.pt"
DEPTH_SCALE = 1000.0  # Orbbec depth in millimeters

# AprilTag localization
APRILTAG_FAMILY = "tagStandard41h12"
APRILTAG_SIZE = 0.0358  # 35.8mm tag

# Base-frame transform (from URDF FK, computed by scripts/compute_T_base_tag.py)
_T_BASE_TAG_PATH = _os.path.join(_os.path.dirname(__file__), "T_base_tag.npy")
T_BASE_TAG = np.load(_T_BASE_TAG_PATH) if _os.path.exists(_T_BASE_TAG_PATH) else None

# Perception grid map parameters (2D occupancy)
GRID_RESOLUTION = 0.025  # meters per cell
GRID_SIZE = 100  # cells (100x100)
GRID_PHYSICAL_SIZE = GRID_SIZE * GRID_RESOLUTION  # 2.5m
GRID_ORIGIN_OFFSET = GRID_PHYSICAL_SIZE / 2  # 1.25m — grid center = robot base
BASE_HEIGHT_THRESHOLD = 0.01  # points below this Z in base frame are floor
ROBOT_RADIUS_CELLS = int(0.15 / GRID_RESOLUTION)  # ~6 cells
TARGET_RADIUS_CELLS = int(0.05 / GRID_RESOLUTION)  # ~2 cells
TARGET_GRID_INFLATE = 1.2  # inflate target circle by this ratio on grid map
