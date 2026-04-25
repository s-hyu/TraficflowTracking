import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


BASE = Path(__file__).resolve().parent
IMAGES_DIR = BASE / "imgs"
# Keep the existing directory name; saved calibration pairs already live here.
POINTS_DIR = BASE / "copmuted_points"
VIS_DIR = BASE / "vis_results"


def parse_args():
    parser = argparse.ArgumentParser("Compute homography from saved point-pair files.")
    parser.add_argument(
        "--pair_file",
        type=str,
        default="point_pairs_left_ground.txt",
        help="Point-pair txt file. Each row is x1 y1 x2 y2.",
    )
    parser.add_argument(
        "--source_name",
        type=str,
        default="left",
        help="Source image name used in output filenames.",
    )
    parser.add_argument(
        "--target_name",
        type=str,
        default="ground",
        help="Target image name used in output filenames.",
    )
    parser.add_argument(
        "--source_image",
        type=str,
        default="left_frame_points30.jpg",
        help="Optional source image for drawing saved source points.",
    )
    parser.add_argument(
        "--target_image",
        type=str,
        default="durlacher_tor_aerial_2022_10cm.png",
        help="Optional target image for drawing saved target points.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=0,
        help="Use only the first N pairs. 0 means use all points.",
    )
    parser.add_argument(
        "--num_train",
        type=int,
        default=0,
        help="Use the first N points for fitting and the remaining points for holdout error evaluation. 0 disables split evaluation.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix appended to H and visualization output filenames.",
    )
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


def load_pairs(pair_file):
    pairs = np.loadtxt(str(pair_file), dtype=np.float64)
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)
    if pairs.shape[1] < 4:
        raise ValueError(f"Pair file must have 4 columns: {pair_file}")
    return pairs


def build_center_ground_pairs():
    left_center_file = POINTS_DIR / "point_pairs.txt"
    left_ground_file = POINTS_DIR / "point_pairs_left_ground.txt"
    if not left_center_file.exists() and (BASE / "point_pairs.txt").exists():
        left_center_file = BASE / "point_pairs.txt"
    if not left_ground_file.exists() and (BASE / "point_pairs_left_ground.txt").exists():
        left_ground_file = BASE / "point_pairs_left_ground.txt"
    if not left_center_file.exists():
        raise FileNotFoundError(f"Missing left-center pair file: {left_center_file}")
    if not left_ground_file.exists():
        raise FileNotFoundError(f"Missing left-ground pair file: {left_ground_file}")

    left_center_pairs = np.loadtxt(str(left_center_file), dtype=np.float64)
    left_ground_pairs = np.loadtxt(str(left_ground_file), dtype=np.float64)
    if left_center_pairs.ndim == 1:
        left_center_pairs = left_center_pairs.reshape(1, -1)
    if left_ground_pairs.ndim == 1:
        left_ground_pairs = left_ground_pairs.reshape(1, -1)

    if left_center_pairs.shape[0] != left_ground_pairs.shape[0]:
        raise ValueError("left-center and left-ground pair counts do not match.")

    center_pts = left_center_pairs[:, 2:4]
    ground_pts = left_ground_pairs[:, 2:4]
    center_ground_pairs = np.hstack([center_pts, ground_pts])

    POINTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = POINTS_DIR / "point_pairs_center_ground.txt"
    np.savetxt(str(output_file), center_ground_pairs, fmt="%.2f")
    return output_file


def draw_points(image_path, points, out_path, color):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image for visualization: {image_path}")

    for idx, (x, y) in enumerate(points):
        px = int(round(float(x)))
        py = int(round(float(y)))
        cv2.circle(image, (px, py), 6, color, -1)
        cv2.putText(
            image,
            str(idx),
            (px + 5, py - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    ok = cv2.imwrite(str(out_path), image)
    if not ok:
        raise RuntimeError(f"Failed to save visualization image: {out_path}")


def evaluate_holdout_error(h_matrix, source_pts, target_pts, target_image_path):
    if len(source_pts) == 0:
        return None

    source_pts = np.asarray(source_pts, dtype=np.float32).reshape(-1, 1, 2)
    target_pts = np.asarray(target_pts, dtype=np.float32).reshape(-1, 2)
    projected = cv2.perspectiveTransform(source_pts, h_matrix).reshape(-1, 2)
    errors = np.linalg.norm(projected - target_pts, axis=1)

    target_image = cv2.imread(str(target_image_path))
    if target_image is None:
        raise FileNotFoundError(f"Failed to load target image for evaluation: {target_image_path}")
    height, width = target_image.shape[:2]
    diagonal = float(np.hypot(width, height))

    return {
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
        "mean_percent_diagonal": float(np.mean(errors) / diagonal * 100.0),
        "num_eval": int(len(errors)),
    }


def main():
    args = parse_args()
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    pair_file = resolve_input_path(args.pair_file, POINTS_DIR)
    if args.pair_file == "point_pairs_center_ground.txt" and not pair_file.exists():
        generated_file = build_center_ground_pairs()
        print(f"Generated pair file: {generated_file}")
        pair_file = generated_file
    if not pair_file.exists():
        raise FileNotFoundError(f"Pair file not found: {pair_file}")

    pairs = load_pairs(pair_file)
    if args.num_points > 0:
        pairs = pairs[:args.num_points]

    source_pts = pairs[:, 0:2]
    target_pts = pairs[:, 2:4]

    fit_source_pts = source_pts
    fit_target_pts = target_pts
    eval_source_pts = np.zeros((0, 2), dtype=np.float64)
    eval_target_pts = np.zeros((0, 2), dtype=np.float64)
    if args.num_train > 0 and args.num_train < len(pairs):
        fit_source_pts = source_pts[:args.num_train]
        fit_target_pts = target_pts[:args.num_train]
        eval_source_pts = source_pts[args.num_train:]
        eval_target_pts = target_pts[args.num_train:]

    h_source_to_target, _ = cv2.findHomography(fit_source_pts, fit_target_pts, 0)
    h_target_to_source = np.linalg.inv(h_source_to_target)

    suffix_parts = []
    if len(fit_source_pts) != len(source_pts):
        suffix_parts.append(f"train{len(fit_source_pts)}")
    if args.output_suffix:
        suffix_parts.append(args.output_suffix.strip("_"))
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    h_source_to_target_path = BASE / f"H_{args.source_name}_to_{args.target_name}{suffix}.txt"
    h_target_to_source_path = BASE / f"H_{args.target_name}_to_{args.source_name}{suffix}.txt"
    if not args.overwrite:
        h_source_to_target_path = non_overwrite_path(h_source_to_target_path)
        h_target_to_source_path = non_overwrite_path(h_target_to_source_path)
    np.savetxt(str(h_source_to_target_path), h_source_to_target, fmt="%.10f")
    np.savetxt(str(h_target_to_source_path), h_target_to_source, fmt="%.10f")

    source_image_path = resolve_input_path(args.source_image, IMAGES_DIR)
    target_image_path = resolve_input_path(args.target_image, IMAGES_DIR)
    vis_suffix = f"_{args.output_suffix.strip('_')}" if args.output_suffix else ""
    source_vis_path = VIS_DIR / f"{source_image_path.stem}_{args.source_name}_points_marked{vis_suffix}.png"
    target_vis_path = VIS_DIR / f"{target_image_path.stem}_{args.target_name}_points_marked{vis_suffix}.png"
    if not args.overwrite:
        source_vis_path = non_overwrite_path(source_vis_path)
        target_vis_path = non_overwrite_path(target_vis_path)
    draw_points(source_image_path, source_pts, source_vis_path, (0, 0, 255))
    draw_points(target_image_path, target_pts, target_vis_path, (255, 0, 0))

    holdout_metrics = evaluate_holdout_error(
        h_source_to_target,
        eval_source_pts,
        eval_target_pts,
        target_image_path,
    )

    print("Used pairs:", len(pairs))
    print("Fit pairs:", len(fit_source_pts))
    print("Eval pairs:", len(eval_source_pts))
    print(f"H ({args.source_name} -> {args.target_name}):")
    print(h_source_to_target)
    print(f"Saved: {h_source_to_target_path}")
    print(f"Saved: {h_target_to_source_path}")
    print(f"Saved: {source_vis_path}")
    print(f"Saved: {target_vis_path}")
    if holdout_metrics is not None:
        print(
            "Holdout reprojection error: "
            f"mean={holdout_metrics['mean_px']:.3f}px, "
            f"max={holdout_metrics['max_px']:.3f}px, "
            f"mean={holdout_metrics['mean_percent_diagonal']:.4f}% of target-image diagonal"
        )


if __name__ == "__main__":
    main()
