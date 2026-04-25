import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2


def parse_args():
    parser = argparse.ArgumentParser("Run SAM2 automatic masks on a video")
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/test2.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="sam_experiments/sam2/checkpoints/sam2.1_hiera_tiny.pt",
        help="SAM2 checkpoint path",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        help="SAM2 model config name",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="sam_experiments/outputs/test2_sam2_auto.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Inference device",
    )
    parser.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    parser.add_argument("--frame_stride", type=int, default=1, help="Process every Nth frame")

    # Mask generation quality/speed trade-offs
    parser.add_argument("--points_per_side", type=int, default=8)
    parser.add_argument("--points_per_batch", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.88)
    parser.add_argument("--stability_score_thresh", type=float, default=0.95)
    parser.add_argument("--box_nms_thresh", type=float, default=0.7)
    parser.add_argument("--max_masks", type=int, default=25, help="Draw at most top-K masks by area")
    parser.add_argument("--min_mask_area", type=int, default=1200, help="Drop tiny masks")
    parser.add_argument(
        "--resize_long_side",
        type=int,
        default=960,
        help="Resize frame long side before SAM2; 0 disables",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_overlay(frame_bgr, anns, max_masks, min_mask_area, rng):
    if not anns:
        return frame_bgr

    anns = sorted(anns, key=lambda x: x["area"], reverse=True)
    canvas = frame_bgr.copy()
    shown = 0
    for ann in anns:
        if shown >= max_masks:
            break
        if ann["area"] < min_mask_area:
            continue

        mask = ann["segmentation"]
        if mask.dtype != np.bool_:
            mask = mask.astype(bool)
        if mask.sum() == 0:
            continue

        color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
        canvas[mask] = (0.55 * canvas[mask] + 0.45 * color).astype(np.uint8)

        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color.tolist(), 1, lineType=cv2.LINE_AA)
        shown += 1

    return canvas


def resize_with_long_side(image, long_side):
    if long_side <= 0:
        return image, 1.0
    h, w = image.shape[:2]
    cur_long = max(h, w)
    if cur_long <= long_side:
        return image, 1.0
    scale = long_side / float(cur_long)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def upscale_annotations(anns, orig_h, orig_w):
    upscaled = []
    for ann in anns:
        seg = ann["segmentation"]
        seg_up = cv2.resize(seg.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
        x, y, w, h = ann["bbox"]
        x = max(0.0, min(float(x), orig_w - 1.0))
        y = max(0.0, min(float(y), orig_h - 1.0))
        w = max(1.0, min(float(w), orig_w - x))
        h = max(1.0, min(float(h), orig_h - y))
        upscaled.append(
            {
                "segmentation": seg_up,
                "bbox": [x, y, w, h],
                "area": int(seg_up.sum()),
            }
        )
    return upscaled


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    video_path = Path(args.video_path)
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Building SAM2 model on {args.device} ...")
    sam2_model = build_sam2(args.model_cfg, str(ckpt_path), device=args.device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        box_nms_thresh=args.box_nms_thresh,
        output_mode="binary_mask",
    )

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

    frame_idx = 0
    processed = 0
    use_amp = args.device == "cuda"
    amp_dtype = torch.bfloat16
    print("[INFO] Start processing video ...")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_idx += 1
        if args.max_frames > 0 and frame_idx > args.max_frames:
            break

        if (frame_idx - 1) % args.frame_stride != 0:
            writer.write(frame_bgr)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        in_rgb, scale = resize_with_long_side(frame_rgb, args.resize_long_side)
        try:
            with torch.inference_mode():
                if use_amp:
                    with torch.autocast("cuda", dtype=amp_dtype):
                        anns = mask_generator.generate(in_rgb)
                else:
                    anns = mask_generator.generate(in_rgb)
        except torch.OutOfMemoryError:
            if args.device != "cuda":
                raise
            torch.cuda.empty_cache()
            print(f"[WARN] OOM at frame {frame_idx}, retry with more aggressive resize.")
            fallback_long = max(512, args.resize_long_side // 2) if args.resize_long_side > 0 else 640
            in_rgb_fb, _ = resize_with_long_side(frame_rgb, fallback_long)
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
                anns = mask_generator.generate(in_rgb_fb)
            scale = in_rgb_fb.shape[0] / float(frame_rgb.shape[0])

        if scale != 1.0:
            anns = upscale_annotations(anns, frame_bgr.shape[0], frame_bgr.shape[1])
        vis = make_overlay(
            frame_bgr,
            anns,
            max_masks=args.max_masks,
            min_mask_area=args.min_mask_area,
            rng=rng,
        )
        writer.write(vis)
        processed += 1
        if use_amp:
            torch.cuda.empty_cache()

        if frame_idx % 10 == 0:
            print(f"[INFO] frame={frame_idx}, processed={processed}, masks={len(anns)}")

    cap.release()
    writer.release()
    print(f"[DONE] Output saved to: {out_path}")


if __name__ == "__main__":
    main()
