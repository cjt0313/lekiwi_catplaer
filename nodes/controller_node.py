"""Controller node: proportional base control + arm forwarding + safety stop."""

import sys
import time
import math

sys.path.insert(0, ".")

from common.config import (
    STATE_PUB, BASE_PLAN_PUB, ARM_PLAN_PUB, FAKE_CAMERA_PUB, CONTROLLER_PUB,
    MAX_BASE_SPEED, MAX_ANGULAR_SPEED, MIN_CAT_DISTANCE, CONTROL_RATE,
    RETRACT_POSE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, make_subscriber, publish, receive, ZmqMessage


KP_LINEAR = 0.5
KP_ANGULAR = 1.0
WAYPOINT_THRESHOLD = 0.1


class ControllerNode:
    def __init__(self):
        self.pub = make_publisher(CONTROLLER_PUB)
        self.state_sub = make_subscriber(STATE_PUB, [MsgType.WORLD_STATE.value])
        self.base_sub = make_subscriber(BASE_PLAN_PUB, [MsgType.BASE_PATH.value])
        self.arm_sub = make_subscriber(ARM_PLAN_PUB, [MsgType.ARM_TARGET.value])
        self.cam_sub = make_subscriber(FAKE_CAMERA_PUB, [MsgType.ROBOT_POSE.value, MsgType.CAT_BBOX_3D.value])

        self.robot_pose = None
        self.cat = None
        self.path = []
        self.waypoint_idx = 0
        self.arm_joints = [0.0] * 6
        self.seq = 0

    def poll(self):
        for _ in range(10):
            msg = receive(self.cam_sub, timeout_ms=1)
            if not msg:
                break
            if msg.header.msg_type == MsgType.ROBOT_POSE.value:
                self.robot_pose = msg.payload
            elif msg.header.msg_type == MsgType.CAT_BBOX_3D.value:
                self.cat = msg.payload

        msg = receive(self.base_sub, timeout_ms=1)
        if msg:
            self.path = msg.payload.get("path", [])
            self.waypoint_idx = 0
            if self.path:
                print(f"[Controller] Received path with {len(self.path)} waypoints")

        msg = receive(self.arm_sub, timeout_ms=1)
        if msg:
            self.arm_joints = msg.payload.get("joint_positions", [0.0] * 6)

        # Drain state_sub to stay current
        receive(self.state_sub, timeout_ms=1)

    def cat_distance(self):
        if not self.robot_pose or not self.cat:
            return float("inf")
        dx = self.cat["center"][0] - self.robot_pose["x"]
        dy = self.cat["center"][1] - self.robot_pose["y"]
        return math.sqrt(dx * dx + dy * dy)

    def compute_command(self):
        safety_stop = self.cat_distance() < MIN_CAT_DISTANCE

        if safety_stop:
            return {
                "base_cmd": {"vx": 0.0, "vy": 0.0, "wz": 0.0},
                "arm_cmd": {"joint_positions": RETRACT_POSE},
                "safety_stop": True,
            }

        vx, vy, wz = 0.0, 0.0, 0.0

        if self.robot_pose and self.path and self.waypoint_idx < len(self.path):
            wp = self.path[self.waypoint_idx]
            dx = wp[0] - self.robot_pose["x"]
            dy = wp[1] - self.robot_pose["y"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < WAYPOINT_THRESHOLD and self.waypoint_idx < len(self.path) - 1:
                self.waypoint_idx += 1
                wp = self.path[self.waypoint_idx]
                dx = wp[0] - self.robot_pose["x"]
                dy = wp[1] - self.robot_pose["y"]
                dist = math.sqrt(dx * dx + dy * dy)

            vx = max(-MAX_BASE_SPEED, min(MAX_BASE_SPEED, KP_LINEAR * dx))
            vy = max(-MAX_BASE_SPEED, min(MAX_BASE_SPEED, KP_LINEAR * dy))

            target_yaw = wp[2] if len(wp) > 2 else math.atan2(dy, dx)
            dyaw = target_yaw - self.robot_pose.get("yaw", 0.0)
            # Normalize angle
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
            wz = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, KP_ANGULAR * dyaw))

        return {
            "base_cmd": {"vx": vx, "vy": vy, "wz": wz},
            "arm_cmd": {"joint_positions": self.arm_joints},
            "safety_stop": False,
            "waypoint_idx": self.waypoint_idx,
        }

    def run(self):
        print("[Controller] Started")
        dt = 1.0 / CONTROL_RATE
        log_count = 0

        while True:
            self.poll()
            cmd = self.compute_command()

            msg = ZmqMessage.create(MsgType.ROBOT_COMMAND, cmd, "controller_node", seq=self.seq)
            publish(self.pub, msg)
            self.seq += 1

            if log_count % 50 == 0:
                bc = cmd["base_cmd"]
                print(f"[Controller] vx={bc['vx']:.3f} vy={bc['vy']:.3f} wz={bc['wz']:.3f} "
                      f"safety={cmd['safety_stop']} wp={self.waypoint_idx}/{len(self.path)}")
            log_count += 1

            time.sleep(dt)


def main():
    time.sleep(0.5)
    node = ControllerNode()
    node.run()


if __name__ == "__main__":
    main()
