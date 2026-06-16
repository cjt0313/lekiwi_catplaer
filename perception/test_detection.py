"""Test YOLO detection on live camera. Saves annotated image with mask + 3D bbox info."""

import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, ".")
from perception.camera import OrbbecCamera
from perception.detection import load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb


def draw_mask_overlay(bgr, mask, color=(0, 255, 0), alpha=0.4):
    overlay = bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)


def draw_bbox_2d(image, mask, color=(0, 255, 0)):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return image
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="cup", help="COCO class name to detect")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--weights", default="yolo11x-seg.pt", help="YOLO model weights")
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.weights}")
    model = load_model(args.weights)

    print("Starting camera...")
    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    for _ in range(5):
        camera.grab()

    print(f"Detecting '{args.target}' (conf >= {args.conf})...")
    rgb, depth = camera.grab()

    if rgb is None:
        print("ERROR: Failed to capture frame")
        camera.stop()
        return

    result = detect_target(model, rgb, args.target, args.conf)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if result is None:
        print(f"'{args.target}' NOT DETECTED. Saving raw image.")
        cv2.putText(bgr, f"No '{args.target}' detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite("test_output_detection.png", bgr)
        camera.stop()
        return

    mask, confidence = result
    print(f"Detected '{args.target}' with confidence={confidence:.3f}")
    print(f"  Mask pixels: {mask.sum()}")

    points = mask_to_3d_points(mask, depth, K, depth_scale=1000.0)
    print(f"  3D points (before outlier removal): {len(points)}")

    if len(points) < 10:
        print("  WARNING: Too few valid depth points!")
        camera.stop()
        return

    points_clean = remove_outliers(points)
    print(f"  3D points (after outlier removal): {len(points_clean)}")

    center, bbox_min, bbox_max = fit_aabb(points_clean)
    size = bbox_max - bbox_min

    print(f"\n  3D Bounding Box (camera frame, meters):")
    print(f"    center:   [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"    bbox_min: [{bbox_min[0]:.3f}, {bbox_min[1]:.3f}, {bbox_min[2]:.3f}]")
    print(f"    bbox_max: [{bbox_max[0]:.3f}, {bbox_max[1]:.3f}, {bbox_max[2]:.3f}]")
    print(f"    size:     [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")

    vis = draw_mask_overlay(bgr, mask, color=(0, 255, 100))
    vis = draw_bbox_2d(vis, mask, color=(0, 255, 0))

    info_lines = [
        f"{args.target} conf={confidence:.2f}",
        f"center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})m",
        f"size: ({size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f})m",
        f"depth: {center[2]:.2f}m",
    ]
    y_offset = 30
    for line in info_lines:
        cv2.putText(vis, line, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25

    cv2.imwrite("test_output_detection.png", vis)
    print(f"\nSaved: test_output_detection.png")
    print(f"  Shows: RGB with green mask overlay, 2D bbox, 3D info text")

    camera.stop()


if __name__ == "__main__":
    main()
