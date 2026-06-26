"""Message types enum."""

from enum import Enum


class MsgType(str, Enum):
    CAT_BBOX_3D = "cat_bbox_3d"
