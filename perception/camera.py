"""Orbbec RGB-D camera wrapper using pyorbbecsdk."""

import ctypes

import numpy as np
from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
)


def _extract_frame_data(raw_buffer, dtype=np.uint8):
    """Extract frame data via ctypes __array_interface__ (pyorbbecsdk requirement)."""
    array_interface = raw_buffer.__array_interface__
    data_ptr, _ = array_interface["data"]
    data_shape = array_interface["shape"]
    dtype_map = {
        np.uint8: (ctypes.c_uint8, 1),
        np.uint16: (ctypes.c_uint16, 2),
    }
    c_type, bytes_per_element = dtype_map[dtype]
    num_elements = data_shape[0] // bytes_per_element
    ctypes_array = (c_type * num_elements).from_address(data_ptr)
    return np.frombuffer(ctypes_array, dtype=dtype).copy()


class OrbbecCamera:
    def __init__(self, color_width=640, color_height=480, color_fps=30):
        self._color_w = color_width
        self._color_h = color_height
        self._color_fps = color_fps
        self._pipeline = None
        self._intrinsics = None

    def start(self):
        self._pipeline = Pipeline()

        config = Config()

        # Find matching color profile
        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None
        for i in range(len(color_profiles)):
            p = color_profiles[i]
            if (p.get_format() == OBFormat.RGB
                    and p.get_width() == self._color_w
                    and p.get_height() == self._color_h):
                color_profile = p
                break
        if color_profile is None:
            raise RuntimeError(
                f"No RGB profile found for {self._color_w}x{self._color_h}")

        # Get HW-aligned depth profile compatible with this color profile
        hw_d2c_profiles = self._pipeline.get_d2c_depth_profile_list(
            color_profile, OBAlignMode.HW_MODE)
        if len(hw_d2c_profiles) == 0:
            raise RuntimeError("No HW D2C depth profiles available")
        depth_profile = hw_d2c_profiles[0]

        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        config.set_align_mode(OBAlignMode.HW_MODE)

        self._pipeline.start(config)

        # Warmup: discard initial frames while sensor stabilizes
        for _ in range(10):
            self._pipeline.wait_for_frames(1000)

        # Use color intrinsics since HW alignment maps depth into color frame
        intrinsic = color_profile.get_intrinsic()
        self._intrinsics = np.array([
            [intrinsic.fx, 0.0, intrinsic.cx],
            [0.0, intrinsic.fy, intrinsic.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        print(f"[OrbbecCamera] Started: color={self._color_w}x{self._color_h}@{color_profile.get_fps()}fps, "
              f"depth=HW-aligned to {self._color_w}x{self._color_h}")
        print(f"[OrbbecCamera] Intrinsics (color): fx={intrinsic.fx:.1f} fy={intrinsic.fy:.1f} "
              f"cx={intrinsic.cx:.1f} cy={intrinsic.cy:.1f}")

    def grab(self, timeout_ms=1000):
        frames = self._pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            return None, None

        c_h, c_w = color_frame.get_height(), color_frame.get_width()
        rgb = _extract_frame_data(color_frame.get_data(), np.uint8).reshape((c_h, c_w, 3))

        d_h, d_w = depth_frame.get_height(), depth_frame.get_width()
        depth = _extract_frame_data(depth_frame.get_data(), np.uint16).reshape((d_h, d_w))

        return rgb, depth

    @property
    def intrinsics(self):
        return self._intrinsics

    def stop(self):
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
            print("[OrbbecCamera] Stopped")
