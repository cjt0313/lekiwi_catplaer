"""Message types enum."""

from enum import Enum


class MsgType(str, Enum):
    ROBOT_POSE = "robot_pose"
    CAT_BBOX_3D = "cat_bbox_3d"
    OCCUPANCY_GRID = "occupancy_grid"
    WORLD_STATE = "world_state"
    PLAY_TARGET = "play_target"
    BASE_PATH = "base_path"
    ARM_TARGET = "arm_target"
    ROBOT_COMMAND = "robot_command"
    ROBOT_STATUS = "robot_status"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
