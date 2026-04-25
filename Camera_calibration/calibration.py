import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "imgs"
# Keep the existing directory name; several saved files already live here.
POINTS_DIR = ROOT / "copmuted_points"

IMAGE_PATHS = {
    "left": IMAGES_DIR / "left_frame_points30.jpg",
    "center": IMAGES_DIR / "center_frame_points30.jpg",
    "ground": IMAGES_DIR / "durlacher_tor_aerial_2022_10cm.png",
}


def parse_args():
    parser = argparse.ArgumentParser("Append calibration point pairs between two images.")
    parser.add_argument("--source", choices=["left", "center", "ground"], default="left")
    parser.add_argument("--target", choices=["left", "center", "ground"], default="ground")
    parser.add_argument(
        "--new_points",
        type=int,
        default=20,
        help="Number of new point pairs to append after existing pairs. 0 means unlimited.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing pair file and start a new pair list.",
    )
    parser.add_argument(
        "--display_width",
        type=int,
        default=1000,
        help="Maximum displayed width per image window. Points are still saved in original image coordinates.",
    )
    parser.add_argument(
        "--display_height",
        type=int,
        default=800,
        help="Maximum displayed height per image window. Points are still saved in original image coordinates.",
    )
    return parser.parse_args()


def load_images():
    images = {}
    for name, path in IMAGE_PATHS.items():
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Failed to load {name} image: {path}")
        images[name] = image
    return images


def load_pair_file(path):
    if not path.exists():
        return [], []

    pairs = np.loadtxt(str(path), dtype=np.float32)
    if pairs.ndim == 1:
        pairs = pairs.reshape(1, -1)
    if pairs.shape[1] < 4:
        raise ValueError(f"Pair file must have at least 4 columns: {path}")

    source_points = pairs[:, 0:2].tolist()
    target_points = pairs[:, 2:4].tolist()
    return source_points, target_points


def draw_labeled_point(image, point, label, color, radius=6):
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    cv2.circle(image, (x, y), radius, color, -1)
    cv2.putText(
        image,
        str(label),
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def fit_scale(image, max_width, max_height):
    height, width = image.shape[:2]
    if max_width <= 0 or max_height <= 0:
        return 1.0
    return min(1.0, max_width / float(width), max_height / float(height))


def to_display_point(point, scale):
    return [float(point[0]) * scale, float(point[1]) * scale]


def to_original_point(x, y, scale):
    return [float(x) / scale, float(y) / scale]


def backup_existing_file(path):
    if not path.exists():
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_backup_{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    print(f"Backed up existing file: {backup_path}")


args = parse_args()
if args.source == args.target:
    raise ValueError("--source and --target must be different.")

POINTS_DIR.mkdir(parents=True, exist_ok=True)

images = load_images()
img_source = images[args.source]
img_target = images[args.target]

source_points_path = POINTS_DIR / f"points_{args.source}_to_{args.target}_{args.source}.txt"
target_points_path = POINTS_DIR / f"points_{args.source}_to_{args.target}_{args.target}.txt"
pair_points_path = POINTS_DIR / f"point_pairs_{args.source}_{args.target}.txt"
h_source_to_target_path = ROOT / f"H_{args.source}_to_{args.target}.txt"
h_target_to_source_path = ROOT / f"H_{args.target}_to_{args.source}.txt"

points_source = []
points_target = []
if not args.fresh:
    points_source, points_target = load_pair_file(pair_points_path)
    if len(points_source) != len(points_target):
        raise ValueError(f"Existing pair file has mismatched point counts: {pair_points_path}")

existing_count = len(points_source)
target_total = existing_count + args.new_points if args.new_points > 0 else None
current_index = existing_count
click_mode = "source"

display_source = img_source.copy()
display_target = img_target.copy()
source_window_name = f"{args.source.capitalize()} Image"
target_window_name = f"{args.target.capitalize()} Image"
source_scale = fit_scale(img_source, args.display_width, args.display_height)
target_scale = fit_scale(img_target, args.display_width, args.display_height)


def redraw_points():
    global display_source, display_target
    display_source = img_source.copy()
    display_target = img_target.copy()

    for i, point in enumerate(points_source):
        color = (120, 120, 120) if i < existing_count else (0, 0, 255)
        draw_labeled_point(display_source, point, i, color)

    for i, point in enumerate(points_target):
        color = (120, 120, 120) if i < existing_count else (255, 0, 0)
        draw_labeled_point(display_target, point, i, color)

    if source_scale != 1.0:
        display_source = cv2.resize(display_source, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA)
    if target_scale != 1.0:
        display_target = cv2.resize(display_target, None, fx=target_scale, fy=target_scale, interpolation=cv2.INTER_AREA)


def reached_target_count():
    return target_total is not None and len(points_source) >= target_total and len(points_target) >= target_total


def undo_last_click():
    global current_index, click_mode

    if len(points_source) <= existing_count and len(points_target) <= existing_count:
        print("No newly added point to undo.")
        return

    if click_mode == "target" and len(points_source) > len(points_target):
        removed = points_source.pop()
        click_mode = "source"
        current_index = len(points_source)
        print(f"Undo pending {args.source.upper()} point {current_index}: {removed}")
        redraw_points()
        return

    if len(points_source) == len(points_target):
        removed_source = points_source.pop()
        removed_target = points_target.pop()
        current_index = len(points_source)
        click_mode = "source"
        print(
            f"Undo pair {current_index}: "
            f"{args.source.upper()} {removed_source}, {args.target.upper()} {removed_target}"
        )
        redraw_points()
        return

    print("Point list state is inconsistent; save is disabled until counts match.")


def click_source(event, x, y, flags, param):
    global click_mode

    if event != cv2.EVENT_LBUTTONDOWN or click_mode != "source":
        return

    if reached_target_count():
        print(f"Already reached target total: {target_total} pairs.")
        return

    point = to_original_point(x, y, source_scale)
    points_source.append(point)
    print(f"[{args.source.upper()}] Point {current_index}: ({point[0]:.2f}, {point[1]:.2f})")
    click_mode = "target"
    redraw_points()


def click_target(event, x, y, flags, param):
    global current_index, click_mode

    if event != cv2.EVENT_LBUTTONDOWN or click_mode != "target":
        return

    if len(points_source) <= len(points_target):
        print(f"Click {args.source.upper()} first for point {current_index}.")
        return

    point = to_original_point(x, y, target_scale)
    points_target.append(point)
    print(f"[{args.target.upper()}] Point {current_index}: ({point[0]:.2f}, {point[1]:.2f})")
    current_index += 1
    click_mode = "source"
    redraw_points()

    if target_total is not None and len(points_target) == target_total:
        print(f"Reached target total: {target_total} pairs. Press 's' to save.")


def save_results():
    if len(points_source) < 4 or len(points_source) != len(points_target):
        print(f"Need at least 4 complete {args.source.upper()}-{args.target.upper()} point pairs.")
        print(f"Current counts: {args.source}={len(points_source)}, {args.target}={len(points_target)}")
        return

    if target_total is not None and len(points_source) != target_total:
        print(f"Warning: expected {target_total} pairs, saving {len(points_source)} pairs.")

    source_array = np.asarray(points_source, dtype=np.float32)
    target_array = np.asarray(points_target, dtype=np.float32)
    pair_array = np.hstack([source_array, target_array])

    for path in (
        source_points_path,
        target_points_path,
        pair_points_path,
        h_source_to_target_path,
        h_target_to_source_path,
    ):
        backup_existing_file(path)

    np.savetxt(str(source_points_path), source_array, fmt="%.2f")
    np.savetxt(str(target_points_path), target_array, fmt="%.2f")
    np.savetxt(str(pair_points_path), pair_array, fmt="%.2f")

    h_source_to_target, _ = cv2.findHomography(source_array, target_array, 0)
    if h_source_to_target is None:
        raise RuntimeError("cv2.findHomography failed. Check point ordering and duplicates.")
    h_target_to_source = np.linalg.inv(h_source_to_target)
    np.savetxt(str(h_source_to_target_path), h_source_to_target, fmt="%.10f")
    np.savetxt(str(h_target_to_source_path), h_target_to_source, fmt="%.10f")

    print(f"Saved {args.source.upper()} points: {source_points_path}")
    print(f"Saved {args.target.upper()} points: {target_points_path}")
    print(f"Saved point pairs: {pair_points_path}")
    print(f"Saved H_{args.source}_to_{args.target}: {h_source_to_target_path}")
    print(f"Saved H_{args.target}_to_{args.source}: {h_target_to_source_path}")
    print(f"Total pairs used for homography: {len(pair_array)}")


redraw_points()
cv2.namedWindow(source_window_name, cv2.WINDOW_NORMAL)
cv2.namedWindow(target_window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(source_window_name, display_source.shape[1], display_source.shape[0])
cv2.resizeWindow(target_window_name, display_target.shape[1], display_target.shape[0])
cv2.setMouseCallback(source_window_name, click_source)
cv2.setMouseCallback(target_window_name, click_target)

print("Instructions:")
if existing_count:
    print(f"Loaded {existing_count} existing pairs from {pair_points_path}.")
else:
    print(f"No existing pair file loaded for {args.source}->{args.target}.")
print(f"Click {args.source.upper()} first, then click {args.target.upper()} for each new pair.")
if target_total is not None:
    print(f"Target total after this run: {target_total} pairs ({args.new_points} new).")
print("Existing points are gray; newly added source points are red; newly added target points are blue.")
print("Press 'u' to undo newly added points, 's' to save and recompute H with all pairs, 'q' to quit.")

while True:
    cv2.imshow(source_window_name, display_source)
    cv2.imshow(target_window_name, display_target)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("u"):
        undo_last_click()
    elif key == ord("s"):
        save_results()
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
