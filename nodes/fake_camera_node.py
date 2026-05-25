"""Fake camera node: publishes simulated RobotPose, CatBBox3D, OccupancyGrid.

Subscribes to RobotCommand to simulate closed-loop motion.
"""

import sys
import time
import random

sys.path.insert(0, ".")

from common.config import (
    FAKE_CAMERA_PUB, CONTROLLER_PUB,
    MAP_RESOLUTION, MAP_WIDTH, MAP_HEIGHT, MAP_ORIGIN,
)
from common.types import MsgType
from common.zmq_message import (
    make_publisher, make_subscriber, publish, receive, ZmqMessage,
)

DT = 0.2  # publish period (5Hz)


def generate_occupancy_grid():
    """Generate a 50x50 grid (2.5m x 2.5m) with walls and obstacles."""
    data = [0] * (MAP_WIDTH * MAP_HEIGHT)

    def set_occupied(r, c):
        if 0 <= r < MAP_HEIGHT and 0 <= c < MAP_WIDTH:
            data[r * MAP_WIDTH + c] = 1

    # Horizontal wall with gap (1 cell thick)
    for c in range(8, 35):
        set_occupied(25, c)
    # Gap in the wall
    for c in range(18, 23):
        data[25 * MAP_WIDTH + c] = 0

    # Vertical wall segment (1 cell thick)
    for r in range(12, 25):
        set_occupied(r, 15)

    # Small box obstacle (upper right area)
    for r in range(36, 39):
        for c in range(36, 39):
            set_occupied(r, c)

    return data


def main():
    pub = make_publisher(FAKE_CAMERA_PUB)
    cmd_sub = make_subscriber(CONTROLLER_PUB, [MsgType.ROBOT_COMMAND.value])
    time.sleep(0.3)

    grid_data = generate_occupancy_grid()
    seq = 0
    robot_x, robot_y, robot_yaw = -0.8, -0.8, 0.0

    print("[FakeCamera] Started, publishing at 5Hz")

    while True:
        # Integrate velocity commands from controller
        msg = receive(cmd_sub, timeout_ms=1)
        if msg and not msg.payload.get("safety_stop", False):
            bc = msg.payload.get("base_cmd", {})
            robot_x += bc.get("vx", 0.0) * DT
            robot_y += bc.get("vy", 0.0) * DT
            robot_yaw += bc.get("wz", 0.0) * DT

        robot_pose_msg = ZmqMessage.create(
            MsgType.ROBOT_POSE,
            {
                "x": robot_x,
                "y": robot_y,
                "yaw": robot_yaw,
                "confidence": 1.0,
            },
            source="fake_camera_node",
            seq=seq,
        )
        publish(pub, robot_pose_msg)

        cat_msg = ZmqMessage.create(
            MsgType.CAT_BBOX_3D,
            {
                "center": [0.5, 0.6, 0.25],
                "bbox_min": [0.3, 0.4, 0.0],
                "bbox_max": [0.7, 0.8, 0.5],
                "confidence": 1.0,
                "visible": True,
            },
            source="fake_camera_node",
            seq=seq,
        )
        publish(pub, cat_msg)

        grid_msg = ZmqMessage.create(
            MsgType.OCCUPANCY_GRID,
            {
                "resolution": MAP_RESOLUTION,
                "width": MAP_WIDTH,
                "height": MAP_HEIGHT,
                "origin": list(MAP_ORIGIN),
                "data": grid_data,
            },
            source="fake_camera_node",
            seq=seq,
        )
        publish(pub, grid_msg)

        if seq % 25 == 0:
            print(f"[FakeCamera] seq={seq} robot=({robot_x:.3f}, {robot_y:.3f}, {robot_yaw:.3f})")

        seq += 1
        time.sleep(DT)


if __name__ == "__main__":
    main()
