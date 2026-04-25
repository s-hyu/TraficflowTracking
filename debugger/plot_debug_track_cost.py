import csv
import math
import os
import os.path as osp

import cv2
import numpy as np


ID_COLUMNS = ("selected_track_id", "best_track_id", "det_index")


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


def resolve_filter(track_id, det_index):
    if track_id is not None:
        return "selected_track_id", track_id
    if det_index is not None:
        return "det_index", det_index
    raise ValueError("Provide --track_id for a tracked object, or --det_index for a per-frame detection index.")


def load_rows(csv_path, id_column, target_id, stage, cost_field):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_column not in reader.fieldnames:
            raise ValueError(f"CSV does not contain id column '{id_column}'. Available: {reader.fieldnames}")
        if cost_field not in reader.fieldnames:
            raise ValueError(f"CSV does not contain cost field '{cost_field}'. Available: {reader.fieldnames}")

        for row in reader:
            if stage and row.get("stage", "") != stage:
                continue
            if to_int(row.get(id_column)) != target_id:
                continue
            x1 = to_float(row.get("x1"))
            y1 = to_float(row.get("y1"))
            w = to_float(row.get("w"))
            h = to_float(row.get("h"))
            cost = to_float(row.get(cost_field))
            frame = to_int(row.get("frame"), -1)
            if frame < 0 or not all(np.isfinite([x1, y1, w, h])):
                continue
            rows.append(
                {
                    "frame": frame,
                    "stage": row.get("stage", ""),
                    "matched": bool(to_int(row.get("matched"), 0)),
                    "selected_track_id": to_int(row.get("selected_track_id"), -1),
                    "x1": x1,
                    "y1": y1,
                    "w": w,
                    "h": h,
                    "cx": x1 + 0.5 * w,
                    "cy": y1 + 0.5 * h,
                    "bev_x": to_float(row.get("bev_x")),
                    "bev_y": to_float(row.get("bev_y")),
                    "cost": cost,
                }
            )
    rows.sort(key=lambda item: item["frame"])
    return rows


def build_track_timeline(rows, track_id):
    """
    Build per-frame trajectory records for one track id.
    - matched frame: real associated detection bbox/point
    - unmatched frame: synthetic missing point at (0, 0)
    """
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
        matched_row = None
        for row in frame_rows:
            if row.get("matched", False) and row.get("selected_track_id", -1) == track_id:
                matched_row = row
                break

        if matched_row is not None:
            item = matched_row.copy()
            item["missing"] = False
        else:
            item = {
                "frame": frame,
                "stage": "",
                "matched": False,
                "selected_track_id": track_id,
                "x1": 0.0,
                "y1": 0.0,
                "w": 0.0,
                "h": 0.0,
                "cx": 0.0,
                "cy": 0.0,
                "bev_x": 0.0,
                "bev_y": 0.0,
                "cost": float("nan"),
                "missing": True,
            }
        timeline.append(item)
    return timeline


def build_cost_timeline(rows):
    """
    Build per-frame cost records:
    - matched frame: use matched row cost
    - unmatched frame: synthetic missing row with cost=0.0
    """
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
            # Keep deterministic choice when multiple matched rows appear.
            best = min(matched_rows, key=lambda x: x.get("cost", float("inf")))
            item = best.copy()
            item["missing"] = False
            if not np.isfinite(item.get("cost", float("nan"))):
                item["cost"] = 0.0
                item["missing"] = True
        else:
            item = {
                "frame": frame,
                "cost": 0.0,
                "missing": True,
            }
        timeline.append(item)
    return timeline


def read_video_size_and_frame(video_path, use_first_frame):
    if not video_path:
        return 0, 0, None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, 0, None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame = None
    if use_first_frame:
        ok, img = cap.read()
        if ok:
            frame = img
    cap.release()
    return width, height, frame


def infer_canvas(rows, video_path="", use_first_frame=False):
    width, height, frame = read_video_size_and_frame(video_path, use_first_frame)
    if width <= 0 or height <= 0:
        max_x = max((row["x1"] + row["w"] for row in rows), default=1280)
        max_y = max((row["y1"] + row["h"] for row in rows), default=720)
        width = int(math.ceil(max_x + 40))
        height = int(math.ceil(max_y + 40))
    if frame is not None:
        return frame.copy()
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_trajectory(rows, output_path, video_path="", use_first_frame=False):
    canvas = infer_canvas(rows, video_path=video_path, use_first_frame=use_first_frame)
    points_info = []
    for row in rows:
        x1, y1, w, h = row["x1"], row["y1"], row["w"], row["h"]
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x1 + w)), int(round(y1 + h)))
        center = (int(round(row["cx"])), int(round(row["cy"])))
        missing = bool(row.get("missing", False))
        points_info.append({"center": center, "missing": missing, "frame": row["frame"]})
        if not missing:
            cv2.rectangle(canvas, pt1, pt2, (210, 210, 210), 1, lineType=cv2.LINE_AA)

    for p0, p1 in zip(points_info[:-1], points_info[1:]):
        if p0["missing"] or p1["missing"]:
            continue
        cv2.line(canvas, p0["center"], p1["center"], (0, 0, 255), 2, lineType=cv2.LINE_AA)

    for idx, info in enumerate(points_info):
        center = info["center"]
        if info["missing"]:
            cv2.line(canvas, (center[0] - 4, center[1] - 4), (center[0] + 4, center[1] + 4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (center[0] - 4, center[1] + 4), (center[0] + 4, center[1] - 4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            color = (0, 180, 0) if idx == 0 else (0, 0, 255)
            cv2.circle(canvas, center, 3, color, -1, lineType=cv2.LINE_AA)

    valid_infos = [info for info in points_info if not info["missing"]]
    if valid_infos:
        start = valid_infos[0]
        end = valid_infos[-1]
        cv2.putText(canvas, f"start f{start['frame']}", (start["center"][0] + 5, start["center"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 0), 2)
        cv2.putText(canvas, f"end f{end['frame']}", (end["center"][0] + 5, end["center"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, canvas)


def draw_bev_trajectory(rows, output_path, ground_image_path=""):
    bev_rows = [
        row for row in rows
        if np.isfinite(row.get("bev_x", float("nan"))) and np.isfinite(row.get("bev_y", float("nan")))
    ]
    if not bev_rows:
        canvas = np.full((720, 1280, 3), 255, dtype=np.uint8)
        cv2.putText(
            canvas,
            "No finite bev_x/bev_y values found. Re-run debugger/demo_track.py to regenerate CSV.",
            (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(output_path, canvas)
        return

    canvas = None
    if ground_image_path:
        canvas = cv2.imread(ground_image_path)
    if canvas is None:
        xs = np.asarray([row["bev_x"] for row in bev_rows], dtype=float)
        ys = np.asarray([row["bev_y"] for row in bev_rows], dtype=float)
        width = int(max(np.max(xs) + 40, 1280))
        height = int(max(np.max(ys) + 40, 720))
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    points_info = [
        {
            "point": (int(round(row["bev_x"])), int(round(row["bev_y"]))),
            "missing": bool(row.get("missing", False)),
            "frame": row["frame"],
        }
        for row in bev_rows
    ]
    for p0, p1 in zip(points_info[:-1], points_info[1:]):
        if p0["missing"] or p1["missing"]:
            continue
        cv2.line(canvas, p0["point"], p1["point"], (255, 0, 0), 2, lineType=cv2.LINE_AA)
    for idx, info in enumerate(points_info):
        point = info["point"]
        if info["missing"]:
            cv2.line(canvas, (point[0] - 4, point[1] - 4), (point[0] + 4, point[1] + 4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, (point[0] - 4, point[1] + 4), (point[0] + 4, point[1] - 4), (0, 0, 255), 2, lineType=cv2.LINE_AA)
        else:
            color = (0, 180, 0) if idx == 0 else (255, 0, 0)
            cv2.circle(canvas, point, 3, color, -1, lineType=cv2.LINE_AA)

    valid_infos = [info for info in points_info if not info["missing"]]
    if not valid_infos:
        cv2.imwrite(output_path, canvas)
        return

    start = valid_infos[0]
    end = valid_infos[-1]
    cv2.putText(canvas, f"BEV trajectory, rows={len(bev_rows)}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(canvas, f"start f{start['frame']}", (start["point"][0] + 5, start["point"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 0), 2)
    cv2.putText(canvas, f"end f{end['frame']}", (end["point"][0] + 5, end["point"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    cv2.imwrite(output_path, canvas)


def draw_cost_curve(rows, output_path, cost_field):
    width, height = 1100, 650
    margin_l, margin_r, margin_t, margin_b = 90, 40, 60, 80
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    frames = [row["frame"] for row in rows]
    costs = [row["cost"] if np.isfinite(row.get("cost", float("nan"))) else 0.0 for row in rows]
    if not frames:
        cv2.putText(canvas, "No finite cost values found", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(output_path, canvas)
        return

    min_f, max_f = min(frames), max(frames)
    min_y = 0.0
    max_y = max(1.0, max(costs))
    if max_y <= min_y:
        max_y = min_y + 1.0

    def map_x(frame):
        if max_f == min_f:
            return margin_l + plot_w // 2
        return int(round(margin_l + (frame - min_f) / (max_f - min_f) * plot_w))

    def map_y(cost):
        return int(round(margin_t + (max_y - cost) / (max_y - min_y) * plot_h))

    # Axes and grid.
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

    cv2.putText(canvas, f"{cost_field} over frames", (margin_l, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(canvas, "frame", (width // 2 - 30, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(canvas, str(min_f), (margin_l - 10, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(canvas, str(max_f), (margin_l + plot_w - 30, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    cv2.imwrite(output_path, canvas)


def main():
    # Edit this block directly when debugging.
    csv_path = "/home/yuhu100/文档/MRT_HiWi/ByteTrack/ByteTrack/debugger/outputs/yolox_x/track_vis/2026_04_20_21_16_22/2026_04_20_21_16_22_debug_costs.csv"
    track_id = 5
    det_index = None
    cost_field = "path_cost"  # path_cost / appearance_cost / position_cost / final_cost
    stage = ""  # primary / secondary / unconfirmed / "" for all
    video_path = "videos/left.mp4"
    ground_image_path = "Camera_calibration/imgs/durlacher_tor_aerial_2022_10cm.png"
    use_first_frame = False
    output_dir = ""  # empty means same folder as csv_path

    id_column, target_id = resolve_filter(track_id=track_id, det_index=det_index)
    rows = load_rows(csv_path, id_column=id_column, target_id=target_id, stage=stage, cost_field=cost_field)
    if not rows:
        raise SystemExit(f"No rows found for {id_column}={target_id} in {csv_path}")
    trajectory_rows = rows
    if track_id is not None:
        trajectory_rows = build_track_timeline(rows, track_id)

    output_dir = output_dir or osp.dirname(osp.abspath(csv_path))
    os.makedirs(output_dir, exist_ok=True)
    stem = osp.splitext(osp.basename(csv_path))[0]
    suffix = f"{stem}_{id_column}_{target_id}_{cost_field}"
    traj_path = osp.join(output_dir, f"{suffix}_trajectory.png")
    bev_traj_path = osp.join(output_dir, f"{suffix}_bev_trajectory.png")
    cost_path = osp.join(output_dir, f"{suffix}_curve.png")

    draw_trajectory(trajectory_rows, traj_path, video_path=video_path, use_first_frame=use_first_frame)
    draw_bev_trajectory(trajectory_rows, bev_traj_path, ground_image_path=ground_image_path)
    cost_rows = build_cost_timeline(rows)
    draw_cost_curve(cost_rows, cost_path, cost_field)
    print(f"rows: {len(rows)}")
    print(f"trajectory: {traj_path}")
    print(f"bev trajectory: {bev_traj_path}")
    print(f"cost curve: {cost_path}")


if __name__ == "__main__":
    main()
