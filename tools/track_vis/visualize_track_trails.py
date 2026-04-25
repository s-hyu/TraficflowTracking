import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser("Visualize track trails from ByteTrack txt result")
    parser.add_argument("--track_file", type=str, required=True, help="tracking txt file in MOT format: frame,id,x1,y1,w,h,...")
    parser.add_argument("--output_image", type=str, required=True, help="output image path")
    parser.add_argument("--video_path", type=str, default="", help="source video path (for size/background)")
    parser.add_argument("--background_image", type=str, default="", help="optional background image path")
    parser.add_argument("--width", type=int, default=0, help="canvas width when video/background is not provided")
    parser.add_argument("--height", type=int, default=0, help="canvas height when video/background is not provided")
    parser.add_argument("--use_background", action="store_true", help="draw trails on video first frame or background image")
    parser.add_argument("--line_thickness", type=int, default=2)
    parser.add_argument("--font_scale", type=float, default=0.6)
    parser.add_argument("--max_ids", type=int, default=0, help="0 means draw all IDs")
    return parser.parse_args()


def _deterministic_color(track_id):
    rng = np.random.default_rng(track_id)
    color = rng.integers(64, 256, size=(3,), dtype=np.uint8)
    return int(color[0]), int(color[1]), int(color[2])


def _load_canvas(args):
    width, height = args.width, args.height
    base_img = None

    if args.background_image:
        bg_path = Path(args.background_image)
        if not bg_path.exists():
            raise FileNotFoundError(f"Background image not found: {bg_path}")
        base_img = cv2.imread(str(bg_path))
        if base_img is None:
            raise RuntimeError(f"Failed to read background image: {bg_path}")
        height, width = base_img.shape[:2]

    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, first = cap.read()
        cap.release()

        if video_w > 0 and video_h > 0:
            width, height = video_w, video_h
        if args.use_background and ok and first is not None:
            base_img = first

    if width <= 0 or height <= 0:
        raise ValueError("Canvas size unknown. Provide --video_path, --background_image, or both --width and --height.")

    if base_img is not None:
        canvas = cv2.resize(base_img, (width, height))
    else:
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    return canvas, width, height


def _read_tracks(track_file):
    tracks = defaultdict(list)
    path = Path(track_file)
    if not path.exists():
        raise FileNotFoundError(f"Track file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
                x1 = float(parts[2])
                y1 = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
            except ValueError:
                continue
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5
            tracks[track_id].append((frame_id, cx, cy))

    for tid in tracks:
        tracks[tid].sort(key=lambda x: x[0])
    return tracks


def draw_trails(canvas, tracks, max_ids, line_thickness, font_scale):
    track_ids = sorted(tracks.keys())
    if max_ids > 0:
        track_ids = track_ids[:max_ids]

    for tid in track_ids:
        pts = tracks[tid]
        if len(pts) == 0:
            continue
        color = _deterministic_color(tid)
        xy = np.array([[int(p[1]), int(p[2])] for p in pts], dtype=np.int32)

        if len(xy) >= 2:
            cv2.polylines(canvas, [xy], isClosed=False, color=color, thickness=line_thickness, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(xy[-1][0]), int(xy[-1][1])), 3, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"ID {tid}",
            (int(xy[-1][0]) + 5, int(xy[-1][1]) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def main():
    args = parse_args()
    count = visualize_track_trails(
        track_file=args.track_file,
        output_image=args.output_image,
        video_path=args.video_path,
        background_image=args.background_image,
        width=args.width,
        height=args.height,
        use_background=args.use_background,
        line_thickness=args.line_thickness,
        font_scale=args.font_scale,
        max_ids=args.max_ids,
    )
    print(f"[INFO] Saved trajectory visualization: {args.output_image}")
    print(f"[INFO] Total IDs drawn: {count}")


def visualize_track_trails(
    track_file,
    output_image,
    video_path="",
    background_image="",
    width=0,
    height=0,
    use_background=False,
    line_thickness=2,
    font_scale=0.6,
    max_ids=0,
):
    args = argparse.Namespace(
        track_file=track_file,
        output_image=output_image,
        video_path=video_path,
        background_image=background_image,
        width=width,
        height=height,
        use_background=use_background,
        line_thickness=line_thickness,
        font_scale=font_scale,
        max_ids=max_ids,
    )
    tracks = _read_tracks(args.track_file)
    canvas, _, _ = _load_canvas(args)
    out = draw_trails(
        canvas=canvas,
        tracks=tracks,
        max_ids=args.max_ids,
        line_thickness=max(1, args.line_thickness),
        font_scale=max(0.1, args.font_scale),
    )

    output_path = Path(args.output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), out)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")
    return len(tracks) if args.max_ids <= 0 else min(args.max_ids, len(tracks))


if __name__ == "__main__":
    main()
