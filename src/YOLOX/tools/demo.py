#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

# import argparse
# import os
# import time
# import subprocess
# import psutil
# from loguru import logger
# import cv2
# import torch

# from yolox.data.data_augment import ValTransform
# from yolox.data.datasets import COCO_CLASSES
# from yolox.exp import get_exp
# from yolox.utils import fuse_model, get_model_info, postprocess, vis

# IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


# def make_parser():
#     parser = argparse.ArgumentParser("YOLOX Demo!")
#     parser.add_argument("demo", default="image", help="demo type: image, video, webcam")
#     parser.add_argument("-expn", "--experiment-name", type=str, default=None)
#     parser.add_argument("-n", "--name", type=str, default=None, help="model name")
#     parser.add_argument("--path", default="./assets/dog.jpg", help="path to images or video")
#     parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
#     parser.add_argument("--save_result", action="store_true", help="save inference result of image/video")
#     parser.add_argument("-f", "--exp_file", default=None, type=str, help="experiment description file")
#     parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint for evaluation")
#     parser.add_argument("--device", default="cpu", type=str, help="device to run model: cpu or gpu")
#     parser.add_argument("--conf", default=0.3, type=float, help="test confidence threshold")
#     parser.add_argument("--nms", default=0.3, type=float, help="test NMS threshold")
#     parser.add_argument("--tsize", default=None, type=int, help="test image size")
#     parser.add_argument("--fp16", default=False, action="store_true", help="use mixed precision")
#     parser.add_argument("--legacy", default=False, action="store_true", help="support older versions")
#     parser.add_argument("--fuse", default=False, action="store_true", help="fuse conv and bn for testing")
#     parser.add_argument("--trt", default=False, action="store_true", help="use TensorRT model for testing")
#     return parser


# class Predictor(object):
#     def __init__(self, model, exp, cls_names=COCO_CLASSES, trt_file=None, decoder=None, device="cpu", fp16=False, legacy=False):
#         self.model = model
#         self.cls_names = cls_names
#         self.decoder = decoder
#         self.num_classes = exp.num_classes
#         self.confthre = exp.test_conf
#         self.nmsthre = exp.nmsthre
#         self.test_size = exp.test_size
#         self.device = device
#         self.fp16 = fp16
#         self.preproc = ValTransform(legacy=legacy)

#     def inference(self, img):
#         img_info = {"id": 0}
#         if isinstance(img, str):
#             img_info["file_name"] = os.path.basename(img)
#             img = cv2.imread(img)
#         else:
#             img_info["file_name"] = None

#         height, width = img.shape[:2]
#         img_info["height"] = height
#         img_info["width"] = width
#         img_info["raw_img"] = img

#         ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
#         img_info["ratio"] = ratio

#         img, _ = self.preproc(img, None, self.test_size)
#         img = torch.from_numpy(img).unsqueeze(0).float()
#         if self.device == "gpu":
#             img = img.cuda()
#             if self.fp16:
#                 img = img.half()  # Use FP16

#         with torch.no_grad():
#             t0 = time.time()
#             outputs = self.model(img)
#             infer_time = time.time() - t0  # Measure inference time
#             fps = 1.0 / infer_time if infer_time > 0 else 0  # Calculate FPS

#             if self.decoder is not None:
#                 outputs = self.decoder(outputs, dtype=outputs.type())
#             outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True)

#             # 🛠️ Log FPS and inference time
#             logger.info(f"Infer time: {infer_time:.4f}s | FPS: {fps:.2f}")

#             # 🛠️ Log GPU details
#             if torch.cuda.is_available() and self.device == "gpu":
#                 num_gpus = torch.cuda.device_count()
#                 current_gpu = torch.cuda.current_device()
#                 gpu_name = torch.cuda.get_device_name(current_gpu)
                
#                 gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
#                 gpu_usage = subprocess.run(
#                     ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
#                     capture_output=True, text=True
#                 )

#                 logger.info(f"Number of GPUs Available: {num_gpus}")
#                 logger.info(f"Using GPU {current_gpu}: {gpu_name}")
#                 logger.info(f"GPU Memory Used: {gpu_memory:.2f} MB | GPU Utilization: {gpu_usage.stdout.strip()}%")

#             # 🛠️ Log CPU and RAM usage
#             cpu_usage = psutil.cpu_percent(interval=1)
#             ram_usage = psutil.virtual_memory().percent
#             logger.info(f"CPU Usage: {cpu_usage:.2f}% | RAM Usage: {ram_usage:.2f}%")

#         return outputs, img_info


# def imageflow_demo(predictor, vis_folder, current_time, args):
#     cap = cv2.VideoCapture(args.path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
    
#     logger.info(f"Processing video: {args.path}")
#     logger.info(f"Video resolution: {width}x{height} | FPS: {fps}")

#     if args.save_result:
#         save_folder = os.path.join(
#             vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
#         )
#         os.makedirs(save_folder, exist_ok=True)
#         save_path = os.path.join(save_folder, os.path.basename(args.path))
#         logger.info(f"Output video will be saved at: {save_path}")

#         vid_writer = cv2.VideoWriter(
#             save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
#         )

#     frame_id = 0
#     while True:
#         ret_val, frame = cap.read()
#         if not ret_val:
#             break  

#         t0 = time.time()  
#         outputs, img_info = predictor.inference(frame)
#         infer_time = time.time() - t0  
#         fps_calc = 1.0 / infer_time if infer_time > 0 else 0  
        
#         logger.info(f"[Frame {frame_id}] Infer time: {infer_time:.4f}s | FPS: {fps_calc:.2f}")
#         frame_id += 1

#     cap.release()
#     if args.save_result:
#         vid_writer.release()
#     logger.info("Video processing completed.")


# def main(exp, args):
#     model = exp.get_model()
#     model.eval()
#     predictor = Predictor(model, exp, COCO_CLASSES, None, None, args.device, args.fp16, args.legacy)
#     current_time = time.localtime()

#     if args.demo == "image":
#         image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
#     elif args.demo == "video":
#         imageflow_demo(predictor, vis_folder, current_time, args)


# if __name__ == "__main__":
#     args = make_parser().parse_args()
#     exp = get_exp(args.exp_file, args.name)
#     main(exp, args)

import argparse
import os
import time
import cv2
import torch
import torch.backends.cudnn as cudnn
import psutil
from loguru import logger

from yolox.exp import get_exp, check_exp_value
from yolox.data.data_augment import ValTransform
from yolox.utils import configure_nccl, configure_omp, configure_module
from yolox.utils.visualize import vis
from yolox.data.datasets import COCO_CLASSES


class Predictor:
    def __init__(self, model, exp, cls_names=COCO_CLASSES, decoder=None, device="cpu", fp16=False, legacy=False):
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

        # ✅ Log GPU/CPU metrics at initialization
        cpu_usage = psutil.cpu_percent(interval=1)
        ram_usage = psutil.virtual_memory().percent
        logger.info(f"CPU Usage: {cpu_usage:.2f}% | RAM Usage: {ram_usage:.2f}%")

        if torch.cuda.is_available() and self.device == "gpu":
            num_gpus = torch.cuda.device_count()
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)  # Total GPU memory in MB

            logger.info(f"Number of GPUs Available: {num_gpus}")
            logger.info(f"Using GPU {current_gpu}: {gpu_name}")
            logger.info(f"Total GPU Memory: {total_memory:.2f} MB")

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
        img = torch.from_numpy(img).unsqueeze(0).float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            infer_time = time.time() - t0
            fps = 1.0 / infer_time if infer_time > 0 else 0

            # ✅ Log inference time and FPS per frame
            logger.info(f"Infer time: {infer_time:.4f}s | FPS: {fps:.2f}")

        return outputs, img_info


def batch_video_demo(predictor, vis_folder, args):
    video_paths = []

    # If input is a directory, get all video files
    if os.path.isdir(args.path):
        video_paths = [os.path.join(args.path, f) for f in os.listdir(args.path) if f.endswith(('.mp4', '.avi', '.mov'))]
    else:
        video_paths = args.path.split(",")

    if not video_paths:
        logger.error("No video files found!")
        return
    
    total_videos = len(video_paths)
    total_frames = 0
    total_latency = 0.0
    total_fps = 0.0
    gpu_usage_list = []

    logger.info(f"Processing {total_videos} videos...")

    for vid_idx, video_path in enumerate(video_paths):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"[{vid_idx+1}/{total_videos}] Processing video: {video_path}")
        logger.info(f"Video resolution: {width}x{height} | FPS: {fps}")

        if args.save_result:
            save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S"))
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, os.path.basename(video_path))
            logger.info(f"Output video will be saved at: {save_path}")

            vid_writer = cv2.VideoWriter(
                save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

        frame_id = 0
        video_latency = 0.0
        video_fps = []

        while True:
            ret_val, frame = cap.read()
            if not ret_val:
                break  

            t0 = time.time()
            outputs, img_info = predictor.inference(frame)
            infer_time = time.time() - t0
            frame_fps = 1.0 / infer_time if infer_time > 0 else 0  

            video_latency += infer_time
            video_fps.append(frame_fps)

            logger.info(f"[Video {vid_idx+1} - Frame {frame_id}] Infer time: {infer_time:.4f}s | FPS: {frame_fps:.2f}")
            frame_id += 1

            if args.save_result:
                vid_writer.write(frame)

        cap.release()
        if args.save_result:
            vid_writer.release()
        
        avg_video_fps = sum(video_fps) / len(video_fps)
        avg_video_latency = (video_latency / len(video_fps)) * 1000  # Convert to ms
        
        logger.info(f"Video {vid_idx+1} completed: Avg FPS: {avg_video_fps:.2f}, Avg Latency: {avg_video_latency:.2f} ms")

        total_frames += len(video_fps)
        total_latency += video_latency
        total_fps += avg_video_fps

        # Track GPU usage in %
        if torch.cuda.is_available() and args.device == "gpu":
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            gpu_usage_percent = (gpu_memory_allocated / gpu_memory_total) * 100
            gpu_usage_list.append(gpu_usage_percent)
    
    avg_fps = total_fps / total_videos
    avg_latency = (total_latency / total_frames) * 1000 if total_frames > 0 else 0
    avg_gpu_usage = sum(gpu_usage_list) / len(gpu_usage_list) if gpu_usage_list else 0

    logger.info("Batch Processing Completed")
    logger.info(f"Total Videos: {total_videos} | Total Frames: {total_frames}")
    logger.info(f"Average FPS: {avg_fps:.2f} | Average Latency: {avg_latency:.2f} ms")
    logger.info(f"Average GPU Usage: {avg_gpu_usage:.2f}%")


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo")
    parser.add_argument("demo", choices=["image", "video"], help="Demo type: 'image' or 'video'")
    parser.add_argument("-n", "--name", type=str, required=True, help="Model name")
    parser.add_argument("-c", "--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--path", type=str, required=True, help="Path to image(s) or video(s)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold for detection")
    parser.add_argument("--nms", type=float, default=0.3, help="Non-maximum suppression threshold")
    parser.add_argument("--tsize", type=int, default=640, help="Test image size")
    parser.add_argument("--save_result", action="store_true", help="Save inference result")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], help="Device to run inference (cpu or gpu)")
    return parser


def main():
    args = make_parser().parse_args()
    exp = get_exp(None, args.name)

    model = exp.get_model()
    if args.device == "gpu":
        model.cuda()
        model.eval()

    predictor = Predictor(model, exp, device=args.device)
    vis_folder = os.path.join(exp.output_dir, args.name, "vis_res")
    os.makedirs(vis_folder, exist_ok=True)

    batch_video_demo(predictor, vis_folder, args)


if __name__ == "__main__":
    configure_module()
    main()
