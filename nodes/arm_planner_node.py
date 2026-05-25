"""Arm planner node: selects predefined arm poses based on state."""

import sys
import time

sys.path.insert(0, ".")

from common.config import (
    STATE_PUB, ARM_PLAN_PUB, MIN_CAT_DISTANCE,
    HOME_POSE, PLAY_POSE, RETRACT_POSE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, make_subscriber, publish, receive, ZmqMessage


class ArmPlannerNode:
    def __init__(self):
        self.pub = make_publisher(ARM_PLAN_PUB)
        self.state_sub = make_subscriber(STATE_PUB, [MsgType.WORLD_STATE.value])
        self.seq = 0
        self.last_pose = None

    def decide_pose(self, world_state):
        state = world_state.get("state")

        if state in ("CAT_TOO_CLOSE", "STOP"):
            pose_name = "RETRACT_POSE"
            joints = RETRACT_POSE
            wand_motion = None
        elif state in ("ARM_READY", "PLAY"):
            pose_name = "PLAY_POSE"
            joints = PLAY_POSE
            wand_motion = {"type": "swing", "amplitude": 0.1, "frequency": 0.5}
        else:
            pose_name = "HOME_POSE"
            joints = HOME_POSE
            wand_motion = None

        if pose_name == self.last_pose:
            return

        payload = {
            "pose_name": pose_name,
            "joint_positions": joints,
            "wand_motion": wand_motion,
        }
        msg = ZmqMessage.create(MsgType.ARM_TARGET, payload, "arm_planner_node", seq=self.seq)
        publish(self.pub, msg)
        self.seq += 1
        self.last_pose = pose_name
        print(f"[ArmPlanner] Pose: {pose_name}")

    def run(self):
        print("[ArmPlanner] Started")
        while True:
            msg = receive(self.state_sub, timeout_ms=100)
            if msg:
                self.decide_pose(msg.payload)


def main():
    time.sleep(0.5)
    node = ArmPlannerNode()
    node.run()


if __name__ == "__main__":
    main()
