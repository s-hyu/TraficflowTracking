import argparse
from datetime import datetime
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

# Make local fast-reid importable when running this script directly in IDE.
REPO_ROOT = Path(__file__).resolve().parents[2]
FASTREID_ROOT = REPO_ROOT / "fast-reid"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(FASTREID_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTREID_ROOT))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.fastreid_extractor import FastReIDExtractor
from yolox.utils import postprocess
from tools.demo_track import THRESHOLD_DEFAULTS, add_maha_region_args
from tools.track_vis.visualize_track_trails import visualize_track_trails


VEHICLE_CLASS_IDS = {0,1,2,3,4,5,7}  # car, motorcycle, bus, truck (COCO)


def parse_args():
    parser = argparse.ArgumentParser("YOLO + SAM2(box prompt) on video")
    parser.add_argument("--video_path", type=str, default="videos/test2.mp4")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.add_argument(
        "--output_path",
        type=str,
        default=f"sam_experiments/outputs/test2_yolo_sam2_tiny_{timestamp}.mp4",
    )

    # YOLO
    parser.add_argument("-f", "--yolo_exp", type=str, default="exps/example/coco/yolox_x.py")
    parser.add_argument("--yolo_ckpt", type=str, default="pretrained/yolox_x.pth")
    parser.add_argument("--yolo_conf", type=float, default=THRESHOLD_DEFAULTS["conf"])
    parser.add_argument("--yolo_nms", type=float, default=THRESHOLD_DEFAULTS["nms"])
    parser.add_argument("--yolo_tsize", type=int, default=None)

    # SAM2 (large by default, as requested)
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default="sam_experiments/sam2/checkpoints/sam2.1_hiera_tiny.pt",
    )
    parser.add_argument(
        "--sam2_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
    )
    parser.add_argument("--sam2_multimask", action="store_true", help="use multimask output and pick best score")

    # Runtime
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--frame_stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--max_boxes", type=int, default=30, help="cap boxes per frame for speed/memory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=30, help="tracker frame rate")
    parser.add_argument("--track_thresh", type=float, default=THRESHOLD_DEFAULTS["track_thresh"])
    parser.add_argument("--track_buffer", type=int, default=THRESHOLD_DEFAULTS["track_buffer"])
    parser.add_argument("--match_thresh", type=float, default=THRESHOLD_DEFAULTS["match_thresh"])
    add_maha_region_args(parser)
    parser.add_argument("--use_maha_gate", dest="use_maha_gate", action="store_true", help="enable Mahalanobis hard gating")
    parser.add_argument("--no_maha_gate", dest="use_maha_gate", action="store_false", help="disable Mahalanobis hard gating")
    parser.set_defaults(use_maha_gate=True)
    parser.add_argument("--mot20", action="store_true")
    parser.add_argument("--reid", action="store_true", help="enable FastReID features for tracker association")
    parser.set_defaults(reid=True)
    parser.add_argument(
        "--reid_config",
        type=str,
        default="fast-reid/configs/Market1501/bagtricks_R50.yml",
        help="FastReID config file",
    )
    parser.add_argument(
        "--reid_weights",
        type=str,
        default="fast-reid/pretrained/market_bot_R50.pth",
        help="FastReID model weights",
    )
    parser.add_argument("--reid_batch", type=int, default=32, help="FastReID batch size")
    parser.add_argument("--reid_device", type=str, default="gpu", choices=["gpu", "cpu"])
    return parser.parse_args()


class YoloPredictor:
    def __init__(self, model, exp, device, fp16=False):
        self.model = model
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=False)

    @torch.no_grad()
    def infer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        ratio = min(self.test_size[0] / h, self.test_size[1] / w)

        img, _ = self.preproc(frame_bgr, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()

        outputs = self.model(img)
        outputs = postprocess(
            outputs,
            self.num_classes,
            self.confthre,
            self.nmsthre,
            class_agnostic_nms=True,
        )
        return outputs, ratio


def filter_vehicle_detections(output):
    if output is None or output.shape[0] == 0:
        return None
    cls_ids = output[:, 6].detach().cpu().numpy().astype(np.int32)
    keep_vehicle = np.array([c in VEHICLE_CLASS_IDS for c in cls_ids], dtype=bool)
    if keep_vehicle.sum() == 0:
        return None
    keep = torch.from_numpy(keep_vehicle).to(output.device)
    return output[keep]


def track_targets_to_boxes(online_targets, max_boxes):
    boxes = []
    track_ids = []
    scores = []
    for t in online_targets:
        tlwh = t.tlwh
        x1 = float(tlwh[0])
        y1 = float(tlwh[1])
        x2 = float(tlwh[0] + tlwh[2])
        y2 = float(tlwh[1] + tlwh[3])
        boxes.append([x1, y1, x2, y2])
        track_ids.append(int(t.track_id))
        scores.append(float(t.score))

    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float32)
    track_ids = np.asarray(track_ids, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float32)

    order = np.argsort(-scores)
    if max_boxes > 0:
        order = order[:max_boxes]
    return boxes[order], track_ids[order], scores[order]


def filter_vehicle_boxes(output, ratio, max_boxes):
    if output is None or output.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    det = output.cpu().numpy()
    cls_ids = det[:, 6].astype(np.int32)
    keep_vehicle = np.array([c in VEHICLE_CLASS_IDS for c in cls_ids], dtype=bool)
    det = det[keep_vehicle]
    if det.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    scores = det[:, 4] * det[:, 5]
    order = np.argsort(-scores)
    if max_boxes > 0:
        order = order[:max_boxes]
    det = det[order]
    scores = scores[order]
    cls_ids = det[:, 6].astype(np.int32)
    boxes = det[:, :4] / ratio
    return boxes.astype(np.float32), cls_ids, scores.astype(np.float32)


def draw_mask_and_box(frame, mask, box_xyxy, color, label):
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    if mask is not None:
        m = mask.astype(bool)
        frame[m] = (0.55 * frame[m] + 0.45 * color).astype(np.uint8)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color.tolist(),
        1,
        cv2.LINE_AA,
    )


def _clip_box_to_image(box_xyxy, h, w):
    x1, y1, x2, y2 = box_xyxy
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w - 1)))
    y2 = int(max(0, min(y2, h - 1)))
    return x1, y1, x2, y2


def _sample_prompt_points_from_box(box_xyxy, h, w):
    x1, y1, x2, y2 = _clip_box_to_image(box_xyxy, h, w)
    if x2 <= x1 or y2 <= y1:
        return None, None

    bw = float(x2 - x1)
    bh = float(y2 - y1)
    point_ratios = np.asarray(
        [
            [0.5, 0.5],
            [0.35, 0.65],
            [0.65, 0.65],
            [0.5, 0.8],
        ],
        dtype=np.float32,
    )
    points = np.empty((len(point_ratios), 2), dtype=np.float32)
    points[:, 0] = x1 + bw * point_ratios[:, 0]
    points[:, 1] = y1 + bh * point_ratios[:, 1]
    points[:, 0] = np.clip(points[:, 0], 0, w - 1)
    points[:, 1] = np.clip(points[:, 1], 0, h - 1)
    labels = np.ones((points.shape[0],), dtype=np.int32)
    return points, labels


def build_masked_patches_for_boxes(frame_bgr, sam2_predictor, tlbrs, multimask_output):
    """
    Build masked BGR patches aligned with tlbrs order.
    - Foreground (mask) keeps original color
    - Background is set to 0
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(frame_rgb)

    patches = []
    for box in tlbrs:
        x1, y1, x2, y2 = _clip_box_to_image(box, h, w)
        if x2 <= x1 or y2 <= y1:
            patches.append(None)
            continue

        base_patch = frame_bgr[y1:y2, x1:x2].copy()
        if base_patch.size == 0:
            patches.append(None)
            continue

        mask_patch = None
        point_coords, point_labels = _sample_prompt_points_from_box((x1, y1, x2, y2), h, w)
        if point_coords is None:
            patches.append(base_patch)
            continue
        try:
            masks, pred_scores, _ = sam2_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            if masks is not None and len(masks) > 0:
                best = int(np.argmax(pred_scores)) if multimask_output else 0
                full_mask = masks[best].astype(bool)
                mask_patch = full_mask[y1:y2, x1:x2]
        except torch.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mask_patch = None

        if mask_patch is None or mask_patch.size == 0 or mask_patch.sum() == 0:
            patches.append(base_patch)
            continue

        fg_patch = np.zeros_like(base_patch)
        fg_patch[mask_patch] = base_patch[mask_patch]
        patches.append(fg_patch)

    return patches


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.device == "gpu" and not torch.cuda.is_available():
        print("[WARN] --device gpu requested but CUDA is unavailable, fallback to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if args.device == "gpu" else "cpu")
    video_path = Path(args.video_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    track_txt_path = out_path.with_suffix(".txt")
    traj_img_path = out_path.with_name(f"{out_path.stem}_traj.png")

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not Path(args.yolo_ckpt).exists():
        raise FileNotFoundError(f"YOLO checkpoint not found: {args.yolo_ckpt}")
    if not Path(args.sam2_ckpt).exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {args.sam2_ckpt}")

    # Build YOLO
    exp = get_exp(args.yolo_exp, None)
    exp.test_conf = args.yolo_conf
    exp.nmsthre = args.yolo_nms
    if args.yolo_tsize is not None:
        exp.test_size = (args.yolo_tsize, args.yolo_tsize)

    yolo_model = exp.get_model().to(device)
    yolo_model.eval()
    ckpt = torch.load(args.yolo_ckpt, map_location="cpu")
    yolo_model.load_state_dict(ckpt["model"])
    yolo_predictor = YoloPredictor(yolo_model, exp, device=device, fp16=args.fp16)
    tracker = BYTETracker(args, frame_rate=args.fps)

    # Build SAM2 image predictor
    sam2_model = build_sam2(args.sam2_cfg, args.sam2_ckpt, device=device.type, apply_postprocessing=False)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    reid_extractor = None
    if args.reid:
        if not Path(args.reid_config).exists():
            raise FileNotFoundError(f"FastReID config not found: {args.reid_config}")
        if not Path(args.reid_weights).exists():
            raise FileNotFoundError(f"FastReID weights not found: {args.reid_weights}")
        reid_device = "cuda" if args.reid_device == "gpu" else "cpu"
        reid_extractor = FastReIDExtractor(
            args.reid_config,
            args.reid_weights,
            device=reid_device,
            batch_size=args.reid_batch,
        )

    id_to_color = {}
    print(f"[INFO] Device: {device}")
    print(f"[INFO] ReID enabled: {args.reid}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_id = 0
    processed = 0
    results = []
    print("[INFO] Running YOLO + SAM2 on video ...")
    with torch.inference_mode():
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_id += 1
            if args.max_frames > 0 and frame_id > args.max_frames:
                break

            if (frame_id - 1) % args.frame_stride != 0:
                writer.write(frame_bgr)
                continue

            outputs, ratio = yolo_predictor.infer(frame_bgr)
            vis = frame_bgr.copy()
            online_targets = []
            vehicle_det = filter_vehicle_detections(outputs[0]) if outputs[0] is not None else None
            if vehicle_det is not None and vehicle_det.shape[0] > 0:
                det_feats = None
                if reid_extractor is not None:
                    tlbrs = vehicle_det[:, :4].detach().cpu().numpy() / ratio
                    masked_patches = build_masked_patches_for_boxes(
                        frame_bgr,
                        sam2_predictor,
                        tlbrs,
                        args.sam2_multimask,
                    )
                    det_feats = reid_extractor.extract_patches(masked_patches)
                online_targets = tracker.update(
                    vehicle_det,
                    [frame_bgr.shape[0], frame_bgr.shape[1]],
                    exp.test_size,
                    det_feats=det_feats,
                )
            for t in online_targets:
                tlwh = t.tlwh
                results.append(
                    f"{frame_id},{int(t.track_id)},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
            boxes, track_ids, scores = track_targets_to_boxes(online_targets, args.max_boxes)

            if boxes.shape[0] > 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                sam2_predictor.set_image(frame_rgb)

                for i in range(boxes.shape[0]):
                    box = boxes[i]
                    tid = int(track_ids[i])
                    score = float(scores[i])
                    if tid not in id_to_color:
                        id_to_color[tid] = rng.integers(0, 255, size=(3,), dtype=np.uint8)
                    color = id_to_color[tid]
                    label = f"ID {tid} {score:.2f}"

                    mask = None
                    point_coords, point_labels = _sample_prompt_points_from_box(box, height, width)
                    if point_coords is None:
                        draw_mask_and_box(vis, mask, box, color, label)
                        continue
                    try:
                        masks, pred_scores, _ = sam2_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=args.sam2_multimask,
                        )
                        if masks is not None and len(masks) > 0:
                            best = int(np.argmax(pred_scores)) if args.sam2_multimask else 0
                            mask = masks[best]
                    except torch.OutOfMemoryError:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                            print(f"[WARN] OOM on frame {frame_id}, box {i}; skip this mask.")
                        else:
                            raise

                    draw_mask_and_box(vis, mask, box, color, label)

            writer.write(vis)
            processed += 1

            if frame_id % 10 == 0:
                print(f"[INFO] frame={frame_id}, processed={processed}, tracks={len(boxes)}")

    cap.release()
    writer.release()
    with open(track_txt_path, "w", encoding="utf-8") as f:
        f.writelines(results)
    visualize_track_trails(
        track_file=str(track_txt_path),
        output_image=str(traj_img_path),
        video_path=str(video_path),
    )
    print(f"[DONE] Saved: {out_path}")
    print(f"[DONE] Saved: {track_txt_path}")
    print(f"[DONE] Saved: {traj_img_path}")


if __name__ == "__main__":
    main()
