"""State node: central state machine, subscribes to perception, publishes WorldState."""

import sys
import time
import math

sys.path.insert(0, ".")

from common.config import (
    FAKE_CAMERA_PUB, STATE_PUB, PLAY_PLAN_PUB, BASE_PLAN_PUB,
    MIN_CAT_DISTANCE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, make_subscriber, publish, receive, ZmqMessage


STATES = [
    "INIT", "WAIT_FOR_ROBOT", "WAIT_FOR_CAT", "SELECT_PLAY_TARGET",
    "NAVIGATE_TO_TARGET", "ARM_READY", "PLAY", "CAT_TOO_CLOSE", "STOP", "ERROR",
]


class StateNode:
    def __init__(self):
        self.pub = make_publisher(STATE_PUB)
        self.cam_sub = make_subscriber(FAKE_CAMERA_PUB)
        self.play_sub = make_subscriber(PLAY_PLAN_PUB, [MsgType.PLAY_TARGET.value])
        self.base_sub = make_subscriber(BASE_PLAN_PUB, [MsgType.BASE_PATH.value])

        self.state = "INIT"
        self.robot_pose = None
        self.cat = None
        self.grid = None
        self.play_target = None
        self.base_path = None
        self.play_start_time = None
        self.seq = 0

    def transition(self, new_state):
        print(f"[StateNode] {self.state} -> {new_state}")
        self.state = new_state

    def publish_world_state(self):
        payload = {
            "state": self.state,
            "robot_pose": self.robot_pose,
            "cat": self.cat,
            "map": self.grid,
        }
        msg = ZmqMessage.create(MsgType.WORLD_STATE, payload, "state_node", seq=self.seq)
        publish(self.pub, msg)
        self.seq += 1

    def cat_distance(self):
        if not self.robot_pose or not self.cat:
            return float("inf")
        dx = self.cat["center"][0] - self.robot_pose["x"]
        dy = self.cat["center"][1] - self.robot_pose["y"]
        return math.sqrt(dx * dx + dy * dy)

    def robot_at_target(self):
        if not self.robot_pose or not self.play_target:
            return False
        dx = self.play_target["target_pose"]["x"] - self.robot_pose["x"]
        dy = self.play_target["target_pose"]["y"] - self.robot_pose["y"]
        return math.sqrt(dx * dx + dy * dy) < 0.15

    def poll_messages(self):
        for _ in range(10):
            msg = receive(self.cam_sub, timeout_ms=1)
            if not msg:
                break
            if msg.header.msg_type == MsgType.ROBOT_POSE.value:
                self.robot_pose = msg.payload
            elif msg.header.msg_type == MsgType.CAT_BBOX_3D.value:
                self.cat = msg.payload
            elif msg.header.msg_type == MsgType.OCCUPANCY_GRID.value:
                self.grid = msg.payload

        msg = receive(self.play_sub, timeout_ms=1)
        if msg:
            self.play_target = msg.payload

        msg = receive(self.base_sub, timeout_ms=1)
        if msg:
            self.base_path = msg.payload

    def step(self):
        self.poll_messages()

        if self.state not in ("STOP", "ERROR") and self.cat_distance() < MIN_CAT_DISTANCE:
            self.transition("CAT_TOO_CLOSE")

        if self.state == "INIT":
            self.transition("WAIT_FOR_ROBOT")

        elif self.state == "WAIT_FOR_ROBOT":
            if self.robot_pose:
                self.transition("WAIT_FOR_CAT")

        elif self.state == "WAIT_FOR_CAT":
            if self.cat and self.cat.get("visible"):
                self.transition("SELECT_PLAY_TARGET")

        elif self.state == "SELECT_PLAY_TARGET":
            if self.play_target:
                self.transition("NAVIGATE_TO_TARGET")

        elif self.state == "NAVIGATE_TO_TARGET":
            if self.robot_at_target():
                self.transition("ARM_READY")

        elif self.state == "ARM_READY":
            self.transition("PLAY")
            self.play_start_time = time.time()

        elif self.state == "PLAY":
            if time.time() - self.play_start_time > 5.0:
                self.play_target = None
                self.base_path = None
                self.transition("SELECT_PLAY_TARGET")

        elif self.state == "CAT_TOO_CLOSE":
            self.transition("STOP")

        self.publish_world_state()

    def run(self):
        print("[StateNode] Started")
        while True:
            self.step()
            time.sleep(0.2)


def main():
    time.sleep(0.5)
    node = StateNode()
    node.run()


if __name__ == "__main__":
    main()
