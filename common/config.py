"""System configuration: ZMQ addresses and parameters."""

# ZMQ PUB addresses (one port per node)
FAKE_CAMERA_PUB = "tcp://127.0.0.1:5560"
STATE_PUB = "tcp://127.0.0.1:5561"
PLAY_PLAN_PUB = "tcp://127.0.0.1:5562"
BASE_PLAN_PUB = "tcp://127.0.0.1:5563"
ARM_PLAN_PUB = "tcp://127.0.0.1:5564"
CONTROLLER_PUB = "tcp://127.0.0.1:5565"

# Robot parameters
DESIRED_CAT_DISTANCE = 1.0  # meters
MIN_CAT_DISTANCE = 0.8  # meters
ROBOT_COLLISION_RADIUS = 0.20  # meters (inflated from 15cm physical)

# Map parameters
MAP_RESOLUTION = 0.05  # meters per cell
MAP_WIDTH = 50  # cells
MAP_HEIGHT = 50  # cells
MAP_ORIGIN = (-1.25, -1.25, 0.0)  # world coords of grid (0,0)

# Control parameters
MAX_BASE_SPEED = 0.2  # m/s
MAX_ANGULAR_SPEED = 0.5  # rad/s
CONTROL_RATE = 10.0  # Hz

# Arm poses [j1, j2, j3, j4, j5, j6]
HOME_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
PLAY_POSE = [0.0, -0.5, 0.8, 0.2, 0.0, 0.3]
RETRACT_POSE = [0.0, -0.2, 0.4, 0.0, 0.0, 0.0]

# Visualization
VIZ_HTTP_PORT = 8080

# Detection parameters
DETECTION_TARGET_CLASS = "cat"
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_MODEL = "yolo11x-seg.pt"
DEPTH_SCALE = 1000.0  # Orbbec depth in millimeters

# AprilTag localization
APRILTAG_FAMILY = "tagStandard41h12"
APRILTAG_SIZE = 0.023  # 23mm tag
