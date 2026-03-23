#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc.

import argparse
import os
import os.path as osp
import time

import cv2
import torch

from loguru import logger

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import get_model_info, postprocess, fuse_model
from yolox.utils.visualize import vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Detector Demo (No Tracking)")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="experiment description file",
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        type=str,
        help="checkpoint file for evaluation",
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result", action="store_true", help="whether to save the result"
    )
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.001, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
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
    parser.add_argument(
        "--dump_dets",
        type=str,
        default=None,
        help="Optional path to dump detector outputs per frame.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def dump_detections(dump_path, frame_id, detections, class_names):
    if detections is None:
        return
    dets = detections
    if hasattr(dets, "detach"):
        dets = dets.detach().cpu().numpy()
    if dets.size == 0:
        return
    lines = []
    for row in dets:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = row[:7]
        score = float(obj_conf * cls_conf)
        w = x2 - x1
        h = y2 - y1
        cls_id_int = int(cls_id)
        cls_name = class_names[cls_id_int] if cls_id_int < len(class_names) else "unknown"
        lines.append(
            f"{frame_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},{cls_id_int},{cls_name}\n"
        )
    with open(dump_path, "a") as f:
        f.writelines(lines)


class DetectorPredictor(object):
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
            from torch2trt import TRTModule  # type: ignore

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
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

    def visualize(self, output, img_info):
        if output is None:
            return img_info["raw_img"]
        output = output.cpu()
        bboxes = output[:, 0:4]
        scores = output[:, 4] * output[:, 5]
        cls = output[:, 6]
        bboxes /= img_info["ratio"]
        return vis(img_info["raw_img"], bboxes, scores, cls, self.confthre, self.cls_names)


def image_demo(predictor, vis_folder, current_time, args):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()
    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path)
        if outputs[0] is not None:
            outputs[0][:, :4] /= img_info["ratio"]
        if args.dump_dets:
            dump_detections(args.dump_dets, frame_id, outputs[0], predictor.cls_names)
        result_image = predictor.visualize(outputs[0], img_info)
        if args.save_result:
            save_folder = osp.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = osp.join(save_folder, osp.basename(img_path))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.save_result:
        save_folder = osp.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = osp.join(save_folder, osp.basename(args.path))
        else:
            save_path = osp.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    frame_id = 0
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            if outputs[0] is not None:
                outputs[0][:, :4] /= img_info["ratio"]
            if args.dump_dets:
                dump_detections(args.dump_dets, frame_id, outputs[0], predictor.cls_names)
            result_frame = predictor.visualize(outputs[0], img_info)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                cv2.namedWindow("yolox-detector", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox-detector", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "det_vis")
        os.makedirs(vis_folder, exist_ok=True)
    else:
        vis_folder = None

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    exp.test_conf = args.conf
    exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
    logger.info("Effective params: conf=%.4f, nms=%.4f", exp.test_conf, exp.nmsthre)

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
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

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

    predictor = DetectorPredictor(
        model, exp, COCO_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy
    )

    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args)
    elif args.demo in ["video", "webcam"]:
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
