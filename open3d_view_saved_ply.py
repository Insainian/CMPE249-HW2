#!/usr/bin/env python3
"""
Simple Open3D viewer for PLY files saved by mmdet3d_inference2.py.

Loads any of the following if present:
- <basename>_points.ply         (PointCloud)
- <basename>_axes.ply           (TriangleMesh)
- <basename>_pred_bboxes.ply    (LineSet)
- <basename>_pred_labels.ply    (LineSet)
- <basename>_gt_bboxes.ply      (LineSet)

Usage:
  python scripts/open3d_view_saved_ply.py --dir /path/to/inference_preview --basename 000008

On macOS, install Open3D via:
  pip install open3d
"""

import argparse
import os
import sys

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d is not installed. Install with `pip install open3d`.\n")
    sys.exit(1)


def load_if_exists(path: str, loader, name: str):
    """Load a geometry with the given loader if the path exists."""
    if os.path.exists(path):
        try:
            obj = loader(path)
            print(f"[LOAD] {name}: {path}")
            return obj
        except Exception as e:
            print(f"[WARN] Failed to load {name} ({path}): {e}")
    else:
        print(f"[SKIP] {name} not found: {path}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for saved PLY outputs")
    parser.add_argument("--dir", default="/Volumes/Samsung_T3/inference_results",
                        help="Folder containing PLY files (default: inference_preview)")
    parser.add_argument("--basename", default="000008",
                        help="Base name, e.g. 000008")
    parser.add_argument("--width", type=int, default=1440,
                        help="Viewer window width (default: 1440)")
    parser.add_argument("--height", type=int, default=900,
                        help="Viewer window height (default: 900)")
    args = parser.parse_args()

    base_dir = os.path.expanduser(args.dir)
    base = args.basename

    points_path = os.path.join(base_dir, f"{base}_points.ply")
    # MODIFIED: Adding prediction path
    pred_path = os.path.join(base_dir, f"{base}_pred.ply")
    axes_path = os.path.join(base_dir, f"{base}_axes.ply")
    pred_bbox_path = os.path.join(base_dir, f"{base}_pred_bboxes.ply")
    pred_label_path = os.path.join(base_dir, f"{base}_pred_labels.ply")
    gt_bbox_path = os.path.join(base_dir, f"{base}_gt_bboxes.ply")

    geoms = []

    # MODIFIED: Added single color to the point cloud for better visibility
    pcd = load_if_exists(points_path, o3d.io.read_point_cloud, "Point cloud")
    if pcd is not None:
        pcd.paint_uniform_color([0.7, 0.7, 0.7])  # light gray
        geoms.append(pcd)

    # MODIFIED: Added prediction point cloud to geom and colored it red
    pred = load_if_exists(pred_path, o3d.io.read_point_cloud, "Predictions")
    if pred is not None:
        pred.paint_uniform_color([1.0, 0.0, 0.0])  # red
        geoms.append(pred)

    axes = load_if_exists(axes_path, o3d.io.read_triangle_mesh, "Coordinate axes")
    if axes is not None:
        geoms.append(axes)

    pred_bboxes = load_if_exists(pred_bbox_path, o3d.io.read_line_set, "Predicted bboxes")
    if pred_bboxes is not None:
        geoms.append(pred_bboxes)

    pred_labels = load_if_exists(pred_label_path, o3d.io.read_line_set, "Predicted labels")
    if pred_labels is not None:
        geoms.append(pred_labels)

    gt_bboxes = load_if_exists(gt_bbox_path, o3d.io.read_line_set, "Ground truth bboxes")
    if gt_bboxes is not None:
        geoms.append(gt_bboxes)

    if not geoms:
        print("\nNo geometries loaded. Check --dir and --basename.")
        print(f"Tried paths:\n  {points_path}\n  {axes_path}\n  {pred_bbox_path}\n  {pred_label_path}\n  {gt_bbox_path}")
        return

    print("\n[INFO] Opening viewer. Controls: mouse to rotate, scroll to zoom, 'Q' to exit.")
    # MODIFIED (till bottom of file): Capturing screenshot automatically
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"PLY Viewer: {base}",
        width=args.width,
        height=args.height,
    )

    for g in geoms:
        vis.add_geometry(g)

    # Render once before capturing
    vis.poll_events()
    vis.update_renderer()

    screenshot_dir = os.path.join(base_dir, "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    png_path = os.path.join(screenshot_dir, f"{base}.png")
    print(f"[INFO] Capturing screenshot to: {png_path}")
    vis.capture_screen_image(png_path, do_render=True)


    # Keep the window open for interactive viewing until you press 'q'
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()