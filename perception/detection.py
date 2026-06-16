"""YOLO-seg detection + depth backprojection + 3D AABB."""

import numpy as np
import cv2
from ultralytics import YOLO


def load_model(weights="yolo11x-seg.pt"):
    model = YOLO(weights)
    return model


def detect_target(model, rgb, target_class, conf_threshold=0.25):
    results = model.predict(source=rgb, conf=conf_threshold, retina_masks=True, verbose=False)
    result = results[0]

    if result.masks is None or result.boxes is None:
        return None

    H, W = rgb.shape[:2]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()

    best_conf = -1.0
    best_mask = None

    for i, cls_id in enumerate(class_ids):
        class_name = result.names[cls_id]
        if class_name == target_class and confs[i] > best_conf:
            best_conf = confs[i]
            best_mask = masks[i]

    if best_mask is None:
        return None

    if best_mask.shape != (H, W):
        best_mask = cv2.resize(best_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    return best_mask > 0.5, float(best_conf)


def mask_to_3d_points(mask, depth, K, depth_scale=1000.0):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    ys, xs = np.where(mask)
    zs = depth[ys, xs].astype(np.float32) / depth_scale

    valid = zs > 0
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    zs = zs[valid]

    X = (xs - cx) * zs / fx
    Y = (ys - cy) * zs / fy
    Z = zs

    return np.stack([X, Y, Z], axis=1)


def remove_outliers(points, q_low=0.05, q_high=0.95):
    if len(points) < 10:
        return points

    lo = np.quantile(points, q_low, axis=0)
    hi = np.quantile(points, q_high, axis=0)

    keep = np.all((points >= lo) & (points <= hi), axis=1)
    return points[keep]


def fit_aabb(points):
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    return center, bbox_min, bbox_max
