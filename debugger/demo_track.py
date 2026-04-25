import argparse
import os
import os.path as osp
import time
import cv2
import torch
from loguru import logger

from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from debugger.tracker.byte_tracker import BYTETracker
from debugger.tracker.fastreid_extractor import FastReIDExtractor
from yolox.tracking_utils.timer import Timer

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from debugger.track_vis.visualize_track_trails import visualize_track_trails

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

"""
Example command:
PYTHONPATH=/home/yuhu100/文档/MRT_HiWi/ByteTrack/ByteTrack/fast-reid:$PYTHONPATH \
python tools/demo_track.py video \
  -f exps/example/coco/yolox_x.py \
  --ckpt pretrained/yolox_x.pth \
  --path videos/left.mp4 \
  --save_result \
  --device gpu \
  --reid \
  --reid_config /home/yuhu100/文档/MRT_HiWi/ByteTrack/ByteTrack/fast-reid/configs/Market1501/bagtricks_R50.yml \
  --reid_weights /home/yuhu100/文档/MRT_HiWi/ByteTrack/ByteTrack/fast-reid/pretrained/market_bot_R50.pth
""" 

THRESHOLD_DEFAULTS = {
    # Detection (YOLO single-frame)
    "conf": 0.080,
    "nms": 0.5,
    # Tracking
    "track_thresh": 0.02,
    "track_buffer": 30,
    "match_thresh": 0.7,
    "maha_thresh": 20.0,
    "maha_thresh_roi": 20.0,
    "maha_roi_polygon": "110,797;107,967;1062,1211;1573,887",
    "velocity_min_speed": 0.05,
    # Post-filter after tracking
    "aspect_ratio_thresh": 100.0,
    "min_box_area": 0.0,
}


def add_maha_region_args(parser):
    parser.add_argument(
        "--maha_thresh",
        type=float,
        default=THRESHOLD_DEFAULTS["maha_thresh"],
        help="squared Mahalanobis gating threshold outside ROI; <=0 disables",
    )
    parser.add_argument(
        "--maha_thresh_roi",
        type=float,
        default=THRESHOLD_DEFAULTS["maha_thresh_roi"],
        help="squared Mahalanobis threshold inside ROI polygon",
    )
    parser.add_argument(
        "--maha_roi_polygon",
        type=str,
        default=THRESHOLD_DEFAULTS["maha_roi_polygon"],
        help="ROI polygon as 'x1,y1;x2,y2;...'; inside uses --maha_thresh_roi",
    )


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument(
        "demo", nargs="?", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        #"--path", default="./datasets/mot/train/MOT17-05-FRCNN/img1", help="path to images or video"
        "--path", default="videos/left.mp4", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="exps/example/coco/yolox_x.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default="pretrained/yolox_x.pth", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # unified thresholds (detection + tracking + post-filter)
    parser.add_argument("--conf", default=THRESHOLD_DEFAULTS["conf"], type=float, help="YOLO detection confidence threshold")
    parser.add_argument("--nms", default=THRESHOLD_DEFAULTS["nms"], type=float, help="YOLO NMS IoU threshold")
    parser.add_argument("--track_thresh", type=float, default=THRESHOLD_DEFAULTS["track_thresh"], help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=THRESHOLD_DEFAULTS["track_buffer"], help="keep lost tracks for this many frames")
    parser.add_argument("--match_thresh", type=float, default=THRESHOLD_DEFAULTS["match_thresh"], help="association threshold for tracking")
    parser.add_argument(
        "--velocity_min_speed",
        type=float,
        default=THRESHOLD_DEFAULTS["velocity_min_speed"],
        help="minimum ground-plane speed for using velocity direction in ReID association",
    )
    add_maha_region_args(parser)
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=THRESHOLD_DEFAULTS["aspect_ratio_thresh"],
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=THRESHOLD_DEFAULTS["min_box_area"], help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--reid", action="store_true", help="enable FastReID features for association")
    parser.add_argument("--reid_config", type=str, default="", help="FastReID config file")
    parser.add_argument("--reid_weights", type=str, default="", help="FastReID model weights")
    parser.add_argument("--reid_batch", type=int, default=32, help="FastReID batch size")
    parser.add_argument("--reid_device", type=str, default="gpu", help="FastReID device: cpu/gpu")
    parser.add_argument(
        "--ground_homography",
        type=str,
        default="auto",
        help=(
            "Homography for bottom-center ground-plane motion. "
            "'auto' maps videos/left.mp4 to H_left_to_ground(_50pts).txt and "
            "videos/center.mp4 to H_center_to_ground(_50pts).txt; empty/none disables."
        ),
    )
    parser.add_argument(
        "--disable_ground_motion",
        action="store_true",
        help="Disable bottom-center ground-plane coordinates in the Kalman motion model.",
    )
    parser.add_argument("--debug_costs", action="store_true", help="enable debugger CSV + overlay for matching costs")
    parser.add_argument("--debug_frame_start", type=int, default=1, help="first frame index for debugger output")
    parser.add_argument("--debug_frame_end", type=int, default=0, help="last frame index for debugger output; 0 means full video")
    parser.add_argument(
        "--debug_track_ids",
        type=str,
        default="",
        help="comma separated track ids to visualize; empty means no track-id filtering",
    )
    return parser


def _parse_debug_track_ids(value):
    if not value:
        return set()
    ids = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.add(int(part))
        except ValueError:
            continue
    return ids


def _save_debug_csv(rows, output_path):
    if not rows:
        return
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            "frame,stage,det_index,matched,selected_track_id,best_track_id,"
            "appearance_cost,position_cost,path_cost,final_cost,x1,y1,w,h,bev_x,bev_y\n"
        )
        for row in rows:
            f.write(
                f"{row['frame']},{row['stage']},{row['det_index']},{int(row['matched'])},"
                f"{row['selected_track_id']},{row['best_track_id']},"
                f"{row['appearance_cost']:.6f},{row['position_cost']:.6f},"
                f"{row['path_cost']:.6f},{row['final_cost']:.6f},"
                f"{row['tlwh'][0]:.2f},{row['tlwh'][1]:.2f},{row['tlwh'][2]:.2f},{row['tlwh'][3]:.2f},"
                f"{row['bev_x']:.6f},{row['bev_y']:.6f}\n"
            )
    logger.info(f"save debug costs to {output_path}")


def _debug_get_color(idx):
    idx = int(idx) * 3
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)


def _build_debug_cost_by_track(rows):
    cost_by_track = {}
    for row in rows or []:
        track_id = row["selected_track_id"] if row["selected_track_id"] >= 0 else row["best_track_id"]
        if track_id is None or track_id < 0:
            continue
        cost_by_track[int(track_id)] = row
    return cost_by_track


def _plot_tracking_debug(image, tlwhs, obj_ids, frame_id=0, fps=0., debug_rows=None):
    im = image.copy()
    text_color = (0, 0, 255)
    text_scale = 2.0
    text_thickness = 2
    line_thickness = 2
    debug_cost_by_track = _build_debug_cost_by_track(debug_rows)

    cv2.putText(
        im,
        "frame: %d fps: %.2f num: %d" % (frame_id, fps, len(tlwhs)),
        (0, int(16 * text_scale)),
        cv2.FONT_HERSHEY_PLAIN,
        text_scale,
        text_color,
        thickness=text_thickness,
    )

    for tlwh, obj_id in zip(tlwhs, obj_ids):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=_debug_get_color(abs(obj_id)), thickness=line_thickness)
        label = str(int(obj_id))
        cost_row = debug_cost_by_track.get(int(obj_id))
        if cost_row is not None:
            label += (
                # f" a={cost_row['appearance_cost']:.2f}"
                # f" p={cost_row['position_cost']:.2f}"
                f" p={cost_row['path_cost']:.2f}"
                # f" f={cost_row['final_cost']:.2f}"
            )
        cv2.putText(
            im,
            label,
            (intbox[0], max(0, intbox[1] - 4)),
            cv2.FONT_HERSHEY_PLAIN,
            text_scale,
            text_color,
            thickness=text_thickness,
        )
    return im


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def _save_track_artifacts(results, vis_folder, timestamp, video_path="", background_image="", width=0, height=0):
    res_file = osp.join(vis_folder, f"{timestamp}.txt")
    with open(res_file, "w", encoding="utf-8") as f:
        f.writelines(results)
    logger.info(f"save results to {res_file}")

    traj_image = osp.join(vis_folder, f"{timestamp}_traj.png")
    visualize_track_trails(
        track_file=res_file,
        output_image=traj_image,
        video_path=video_path,
        background_image=background_image,
        width=width,
        height=height,
        use_background=bool(background_image),
    )
    logger.info(f"save trajectory image to {traj_image}")


def filter_to_first_n_coco_classes(detections, n=8):
    """Keep only detections whose COCO class id is in [0, n)."""
    if detections is None:
        return None
    keep_mask = detections[:, 6] < n
    filtered = detections[keep_mask]
    if filtered.shape[0] == 0:
        return None
    return filtered


class OfficialPredictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule # type: ignore

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img, timer=None):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if isinstance(self.device, torch.device) and self.device.type == "cuda":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            # t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs,
                self.num_classes,
                self.confthre,
                self.nmsthre,
                class_agnostic_nms=True,
            )

        return outputs, img_info
    

def image_demo(predictor, vis_folder, current_time, args, reid_extractor=None):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    image_width = 0
    image_height = 0

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        image_width = img_info["width"]
        image_height = img_info["height"]
        det_feats = None
        if outputs[0] is not None:
            outputs[0] = filter_to_first_n_coco_classes(outputs[0], n=8)
        if reid_extractor is not None and outputs[0] is not None and outputs[0].shape[0] > 0:
            tlbrs = outputs[0][:, :4].cpu().numpy()
            tlbrs = tlbrs / img_info["ratio"]
            det_feats = reid_extractor.extract(img_info["raw_img"], tlbrs)
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, det_feats)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            online_im = _plot_tracking_debug(
                img_info['raw_img'],
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1. / timer.average_time,
                debug_rows=tracker.debug_last_rows if args.debug_costs else None,
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    bg_image = args.path if osp.isfile(args.path) else ""
    _save_track_artifacts(
        results=results,
        vis_folder=save_folder,
        timestamp=timestamp,
        video_path="",
        background_image=bg_image,
        width=int(image_width),
        height=int(image_height),
    )


def imageflow_demo(predictor, vis_folder, current_time, args, reid_extractor=None):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = BYTETracker(args, frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    debug_rows = []
    canvas_w = int(width)
    canvas_h = int(height)
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            canvas_w = int(img_info["width"])
            canvas_h = int(img_info["height"])
            det_feats = None
            if outputs[0] is not None:
                outputs[0] = filter_to_first_n_coco_classes(outputs[0], n=8)
            if reid_extractor is not None and outputs[0] is not None and outputs[0].shape[0] > 0:
                tlbrs = outputs[0][:, :4].cpu().numpy()
                tlbrs = tlbrs / img_info["ratio"]
                det_feats = reid_extractor.extract(img_info["raw_img"], tlbrs)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, det_feats)
                if args.debug_costs and tracker.debug_last_rows:
                    debug_rows.extend(tracker.debug_last_rows)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = _plot_tracking_debug(
                    img_info['raw_img'],
                    online_tlwhs,
                    online_ids,
                    frame_id=frame_id + 1,
                    fps=1. / timer.average_time,
                    debug_rows=tracker.debug_last_rows if args.debug_costs else None,
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            if args.debug_costs and args.debug_frame_end > 0 and (frame_id + 1) >= args.debug_frame_end:
                break
        else:
            break
        frame_id += 1

    cap.release()
    vid_writer.release()
    video_path = args.path if args.demo == "video" else ""
    _save_track_artifacts(
        results=results,
        vis_folder=save_folder,
        timestamp=timestamp,
        video_path=video_path,
        background_image="",
        width=canvas_w,
        height=canvas_h,
    )
    if args.debug_costs:
        _save_debug_csv(debug_rows, osp.join(save_folder, f"{timestamp}_debug_costs.csv"))


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    if args.demo in {"image", "video"} and not args.path:
        raise ValueError("Please provide --path for image/video input, or use the matching launch.json configuration.")

    output_dir = osp.join("debugger", "outputs", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = OfficialPredictor(model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    reid_extractor = None
    if args.reid:
        if not args.reid_config or not args.reid_weights:
            raise ValueError("FastReID enabled but reid_config/reid_weights not provided.")
        reid_device = "cuda" if args.reid_device == "gpu" else "cpu"
        reid_extractor = FastReIDExtractor(
            args.reid_config,
            args.reid_weights,
            device=reid_device,
            batch_size=args.reid_batch,
        )
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args, reid_extractor=reid_extractor)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args, reid_extractor=reid_extractor)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
