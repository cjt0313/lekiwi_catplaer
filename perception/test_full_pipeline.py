"""Full pipeline test: continuous detection with live visualization output."""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, ".")
from perception.camera import OrbbecCamera
from perception.detection import load_model, detect_target, mask_to_3d_points, remove_outliers, fit_aabb


def colorize_depth(depth, max_depth_mm=3000):
    valid = depth > 0
    d = depth.astype(np.float32)
    d[~valid] = 0
    d = np.clip(d / max_depth_mm, 0, 1)
    colored = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored


def main():
    target_class = sys.argv[1] if len(sys.argv) > 1 else "cup"
    weights = sys.argv[2] if len(sys.argv) > 2 else "yolo11x-seg.pt"

    print(f"Full pipeline test: target='{target_class}', model={weights}")
    model = load_model(weights)

    camera = OrbbecCamera()
    camera.start()
    K = camera.intrinsics

    # Warm up
    for _ in range(5):
        camera.grab()

    frame_count = 0
    fps_start = time.time()
    save_interval = 15  # save visualization every N frames

    print("Running... Press Ctrl+C to stop. Visualization saved periodically.")

    try:
        while True:
            rgb, depth = camera.grab()
            if rgb is None:
                continue

            t0 = time.time()
            result = detect_target(model, rgb, target_class, conf_threshold=0.25)
            inference_ms = (time.time() - t0) * 1000

            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth_vis = colorize_depth(depth)

            if result is not None:
                mask, confidence = result
                points = mask_to_3d_points(mask, depth, K, depth_scale=1000.0)
                points = remove_outliers(points)

                # Draw mask overlay
                overlay = bgr.copy()
                overlay[mask] = (0, 255, 100)
                bgr = cv2.addWeighted(overlay, 0.4, bgr, 0.6, 0)

                if len(points) >= 10:
                    center, bbox_min, bbox_max = fit_aabb(points)
                    size = bbox_max - bbox_min

                    # Draw 2D bbox
                    ys, xs = np.where(mask)
                    cv2.rectangle(bgr, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 2)

                    # Info text
                    lines = [
                        f"{target_class} {confidence:.2f}",
                        f"Z={center[2]:.2f}m",
                        f"size=({size[0]:.2f},{size[1]:.2f},{size[2]:.2f})",
                    ]
                    for i, line in enumerate(lines):
                        cv2.putText(bgr, line, (xs.min(), ys.min() - 10 - i * 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Also draw depth points on depth vis
                    depth_vis[mask] = (0, 255, 0)

                    status = f"DETECTED: center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
                else:
                    status = "DETECTED but too few depth points"
            else:
                status = "No detection"

            # FPS counter
            frame_count += 1
            elapsed = time.time() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            cv2.putText(bgr, f"FPS: {fps:.1f}  Inference: {inference_ms:.0f}ms", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(bgr, status, (10, bgr.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if frame_count % save_interval == 0:
                composite = np.hstack([bgr, depth_vis])
                cv2.imwrite("test_output_pipeline.png", composite)

            if frame_count % 30 == 0:
                print(f"  frame={frame_count} fps={fps:.1f} inference={inference_ms:.0f}ms {status}")

    except KeyboardInterrupt:
        pass

    # Save final frame
    composite = np.hstack([bgr, depth_vis])
    cv2.imwrite("test_output_pipeline.png", composite)
    print(f"\nFinal visualization saved: test_output_pipeline.png")
    print(f"Total frames: {frame_count}, avg FPS: {frame_count / elapsed:.1f}")

    camera.stop()


if __name__ == "__main__":
    main()
