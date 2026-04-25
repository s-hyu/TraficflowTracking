import csv
import os
import os.path as osp

import cv2
import numpy as np


def to_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=float("nan")):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_track_costs(csv_path, track_id, stage=""):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"frame", "stage", "selected_track_id", "appearance_cost"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")

        for row in reader:
            if stage and row.get("stage", "") != stage:
                continue
            if to_int(row.get("selected_track_id")) != track_id:
                continue
            frame = to_int(row.get("frame"), -1)
            cost = to_float(row.get("appearance_cost"))
            matched = bool(to_int(row.get("matched"), 0))
            if frame < 0:
                continue
            rows.append({"frame": frame, "appearance_cost": cost, "matched": matched})

    rows.sort(key=lambda item: item["frame"])
    return rows


def build_cost_timeline(rows):
    """Per-frame appearance cost timeline with missing frames marked."""
    if not rows:
        return []

    by_frame = {}
    for row in rows:
        by_frame.setdefault(row["frame"], []).append(row)

    frame_min = min(by_frame.keys())
    frame_max = max(by_frame.keys())
    timeline = []
    for frame in range(frame_min, frame_max + 1):
        frame_rows = by_frame.get(frame, [])
        matched_rows = [row for row in frame_rows if row.get("matched", False)]
        if matched_rows:
            best = min(matched_rows, key=lambda x: x.get("appearance_cost", float("inf")))
            cost = best.get("appearance_cost", float("nan"))
            if np.isfinite(cost):
                timeline.append({"frame": frame, "appearance_cost": float(cost), "missing": False})
            else:
                timeline.append({"frame": frame, "appearance_cost": 0.0, "missing": True})
        else:
            timeline.append({"frame": frame, "appearance_cost": 0.0, "missing": True})
    return timeline


def draw_curve(rows, output_path, track_id):
    width, height = 1100, 650
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 80
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    if not rows:
        cv2.putText(canvas, f"No appearance cost rows for track_id={track_id}", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(output_path, canvas)
        return

    frames = [row["frame"] for row in rows]
    costs = [row["appearance_cost"] if np.isfinite(row.get("appearance_cost", float("nan"))) else 0.0 for row in rows]
    min_f, max_f = min(frames), max(frames)
    min_y = 0.0
    max_y = max(1.0, max(costs))

    def map_x(frame):
        if max_f == min_f:
            return margin_l + plot_w // 2
        return int(round(margin_l + (frame - min_f) / (max_f - min_f) * plot_w))

    def map_y(cost):
        return int(round(margin_t + (max_y - cost) / (max_y - min_y) * plot_h))

    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (0, 0, 0), 1)
    for i in range(6):
        y_val = min_y + (max_y - min_y) * i / 5.0
        y = map_y(y_val)
        cv2.line(canvas, (margin_l, y), (margin_l + plot_w, y), (230, 230, 230), 1)
        cv2.putText(canvas, f"{y_val:.2f}", (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    points_info = []
    for row, cost in zip(rows, costs):
        pt = (map_x(row["frame"]), map_y(cost))
        points_info.append({"pt": pt, "missing": bool(row.get("missing", False))})

    for p0, p1 in zip(points_info[:-1], points_info[1:]):
        if p0["missing"] or p1["missing"]:
            continue
        cv2.line(canvas, p0["pt"], p1["pt"], (0, 0, 255), 2, lineType=cv2.LINE_AA)

    for info in points_info:
        x, y = info["pt"]
        if info["missing"]:
            cv2.line(canvas, (x - 4, y - 4), (x + 4, y + 4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (x - 4, y + 4), (x + 4, y - 4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            cv2.circle(canvas, (x, y), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)

    cv2.putText(canvas, f"appearance_cost over frames, track_id={track_id}", (margin_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(canvas, "frame", (width // 2 - 30, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, str(min_f), (margin_l - 10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(canvas, str(max_f), (margin_l + plot_w - 30, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.imwrite(output_path, canvas)


def main():
    # Edit this block directly when debugging.
    csv_path = "/home/yuhu100/文档/MRT_HiWi/ByteTrack/ByteTrack/debugger/outputs/yolox_x/track_vis/2026_04_20_21_16_22/2026_04_20_21_16_22_debug_costs.csv"
    track_id = 5
    stage = ""  # primary / secondary / unconfirmed / "" for all
    output_dir = ""  # empty means same folder as csv_path

    rows = load_track_costs(csv_path, track_id=track_id, stage=stage)
    rows = build_cost_timeline(rows)
    output_dir = output_dir or osp.dirname(osp.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    stem = osp.splitext(osp.basename(csv_path))[0]
    output_path = osp.join(output_dir, f"{stem}_track_{track_id}_appearance_cost_curve.png")
    draw_curve(rows, output_path, track_id=track_id)
    print(f"rows: {len(rows)}")
    print(f"appearance curve: {output_path}")


if __name__ == "__main__":
    main()
