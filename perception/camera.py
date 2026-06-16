"""Orbbec RGB-D camera wrapper using pyorbbecsdk."""

import numpy as np
from pyorbbecsdk import (
    Pipeline, Config, OBStreamType, OBFormat, OBAlignMode,
)


class OrbbecCamera:
    def __init__(self, color_width=640, color_height=480, color_fps=30,
                 depth_width=640, depth_height=480, depth_fps=30):
        self._color_w = color_width
        self._color_h = color_height
        self._color_fps = color_fps
        self._depth_w = depth_width
        self._depth_h = depth_height
        self._depth_fps = depth_fps
        self._pipeline = None
        self._intrinsics = None

    def start(self):
        self._pipeline = Pipeline()

        config = Config()

        color_profiles = self._pipeline.get_stream_profile_list(OBStreamType.COLOR_STREAM)
        color_profile = color_profiles.get_video_stream_profile(
            self._color_w, self._color_h, OBFormat.RGB, self._color_fps
        )
        config.enable_stream(color_profile)

        depth_profiles = self._pipeline.get_stream_profile_list(OBStreamType.DEPTH_STREAM)
        depth_profile = depth_profiles.get_video_stream_profile(
            self._depth_w, self._depth_h, OBFormat.Y16, self._depth_fps
        )
        config.enable_stream(depth_profile)

        config.set_align_mode(OBAlignMode.HW_MODE)

        self._pipeline.start(config)

        intrinsic = depth_profile.get_intrinsic()
        self._intrinsics = np.array([
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        print(f"[OrbbecCamera] Started: color={self._color_w}x{self._color_h}@{self._color_fps}fps, "
              f"depth={self._depth_w}x{self._depth_h}@{self._depth_fps}fps")
        print(f"[OrbbecCamera] Intrinsics: fx={intrinsic.fx:.1f} fy={intrinsic.fy:.1f} "
              f"cx={intrinsic.cx:.1f} cy={intrinsic.cy:.1f}")

    def grab(self, timeout_ms=1000):
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        rgb = np.asarray(color_frame.get_data()).reshape(
            (color_frame.get_height(), color_frame.get_width(), 3)
        )

        depth = np.asarray(depth_frame.get_data()).reshape(
            (depth_frame.get_height(), depth_frame.get_width())
        ).astype(np.uint16)

        return rgb, depth

    @property
    def intrinsics(self):
        return self._intrinsics

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            print("[OrbbecCamera] Stopped")
