"""Real camera node: captures RGB-D from Orbbec, detects target, publishes CAT_BBOX_3D."""

import sys
import time

sys.path.insert(0, ".")

from common.config import (
    FAKE_CAMERA_PUB, DETECTION_TARGET_CLASS,
    DETECTION_CONF_THRESHOLD, DETECTION_MODEL, DEPTH_SCALE,
)
from common.types import MsgType
from common.zmq_message import make_publisher, publish, ZmqMessage
from perception.camera import OrbbecCamera
from perception.detection import load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb


class RealCameraNode:
    def __init__(self):
        self.pub = make_publisher(FAKE_CAMERA_PUB)
        self.camera = OrbbecCamera()
        self.model = None
        self.seq = 0

    def start(self):
        print(f"[RealCamera] Loading model: {DETECTION_MODEL}")
        self.model = load_model(DETECTION_MODEL)

        print("[RealCamera] Starting camera...")
        self.camera.start()

        # Warm up: skip first few frames
        for _ in range(5):
            self.camera.grab()

        print("[RealCamera] Ready. Publishing CAT_BBOX_3D on", FAKE_CAMERA_PUB)

    def step(self):
        rgb, depth = self.camera.grab()
        if rgb is None:
            return

        K = self.camera.intrinsics
        result = detect_target(self.model, rgb, DETECTION_TARGET_CLASS, DETECTION_CONF_THRESHOLD)

        if result is not None:
            mask, confidence = result
            points = mask_to_3d_points(mask, depth, K, depth_scale=DEPTH_SCALE)
            points = remove_outliers(points)

            if len(points) >= 10:
                center, bbox_min, bbox_max = fit_aabb(points)
                payload = {
                    "center": center.tolist(),
                    "bbox_min": bbox_min.tolist(),
                    "bbox_max": bbox_max.tolist(),
                    "confidence": confidence,
                    "visible": True,
                }
            else:
                payload = {
                    "center": None,
                    "bbox_min": None,
                    "bbox_max": None,
                    "confidence": 0.0,
                    "visible": False,
                }
        else:
            payload = {
                "center": None,
                "bbox_min": None,
                "bbox_max": None,
                "confidence": 0.0,
                "visible": False,
            }

        msg = ZmqMessage.create(MsgType.CAT_BBOX_3D, payload, "real_camera_node",
                                seq=self.seq, frame_id="camera")
        publish(self.pub, msg)
        self.seq += 1

        if self.seq % 10 == 0:
            if payload["visible"]:
                c = payload["center"]
                print(f"[RealCamera] seq={self.seq} DETECTED center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                      f"conf={payload['confidence']:.2f}")
            else:
                print(f"[RealCamera] seq={self.seq} no detection")

    def run(self):
        self.start()
        try:
            while True:
                self.step()
        except KeyboardInterrupt:
            print("\n[RealCamera] Shutting down...")
        finally:
            self.camera.stop()


def main():
    node = RealCameraNode()
    node.run()


if __name__ == "__main__":
    main()
