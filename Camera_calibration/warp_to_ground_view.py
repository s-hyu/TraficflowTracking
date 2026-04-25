import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
IMAGES_DIR = BASE / "imgs"
VIS_DIR = BASE / "vis_results"


def parse_args():
    parser = argparse.ArgumentParser("Warp a camera image into the ground-view plane using a homography.")
    parser.add_argument("--source_image", type=str, default="left_frame_points30.jpg")
    parser.add_argument("--target_image", type=str, default="durlacher_tor_aerial_2022_10cm.png")
    parser.add_argument("--homography", type=str, default="H_left_to_ground.txt")
    parser.add_argument("--output_warp", type=str, default="left_to_ground_warp.png")
    parser.add_argument("--output_overlay", type=str, default="left_to_ground_overlay.png")
    parser.add_argument("--overlay_alpha", type=float, default=0.55)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files. By default, existing files get a timestamped suffix.",
    )
    return parser.parse_args()


def resolve_input_path(path_str, fallback_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path
    direct = BASE / path
    if direct.exists():
        return direct
    fallback = fallback_dir / path
    if fallback.exists():
        return fallback
    return fallback


def resolve_output_path(path_str, output_dir):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return output_dir / path


def non_overwrite_path(path):
    if not path.exists():
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def main():
    args = parse_args()
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    source_path = resolve_input_path(args.source_image, IMAGES_DIR)
    target_path = resolve_input_path(args.target_image, IMAGES_DIR)
    homography_path = resolve_input_path(args.homography, BASE)

    source = cv2.imread(str(source_path))
    target = cv2.imread(str(target_path))
    if source is None:
        raise FileNotFoundError(f"Failed to load source image: {source_path}")
    if target is None:
        raise FileNotFoundError(f"Failed to load target image: {target_path}")
    if not homography_path.exists():
        raise FileNotFoundError(f"Homography file not found: {homography_path}")

    h_matrix = np.loadtxt(str(homography_path), dtype=np.float64)
    target_h, target_w = target.shape[:2]
    warped = cv2.warpPerspective(source, h_matrix, (target_w, target_h))

    alpha = float(np.clip(args.overlay_alpha, 0.0, 1.0))
    overlay = cv2.addWeighted(warped, alpha, target, 1.0 - alpha, 0.0)

    output_warp = resolve_output_path(args.output_warp, VIS_DIR)
    output_overlay = resolve_output_path(args.output_overlay, VIS_DIR)
    if not args.overwrite:
        output_warp = non_overwrite_path(output_warp)
        output_overlay = non_overwrite_path(output_overlay)
    if not cv2.imwrite(str(output_warp), warped):
        raise RuntimeError(f"Failed to save warped image: {output_warp}")
    if not cv2.imwrite(str(output_overlay), overlay):
        raise RuntimeError(f"Failed to save overlay image: {output_overlay}")

    print(f"Saved warped image: {output_warp}")
    print(f"Saved overlay image: {output_overlay}")


if __name__ == "__main__":
    main()
