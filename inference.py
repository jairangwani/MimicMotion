import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image


from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

# This constant is likely defined in your project. If not, you may need to define it.
# For example: ASPECT_RATIO = 9/16 or 16/9
try:
    from constants import ASPECT_RATIO
except ImportError:
    ASPECT_RATIO = 576 / 1024
    print("Warning: 'constants.py' not found. Using default ASPECT_RATIO.")


from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
# This script now depends on the modified preprocess.py file
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(video_path, image_path, resolution=576, sample_stride=2, num_frames=None):
    """
    Preprocesses the reference image and extracts poses from the driving video.

    Args:
        video_path (str): Path to the input driving video.
        image_path (str): Path to the reference image.
        resolution (int, optional): Target resolution. Defaults to 576.
        sample_stride (int, optional): Frame sampling rate. Defaults to 2.
        num_frames (int, optional): The exact number of frames to extract from the video.
                                    Defaults to None (process all frames).
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]

    # Compute target h/w according to original aspect ratio
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    
    # Get image & video pose using the functions from the modified preprocess.py
    image_pose = get_image_pose(image_pixels)
    
    # Pass the `num_frames` parameter to the pose extraction function
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride, num_frames=num_frames)
    
    logger.info(f"Successfully extracted {video_pose.shape[0]} pose frames from the video.")
    
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    
    # Normalize pixels for the model
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1


def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    
    # The `num_frames` parameter for the pipeline correctly uses the length of the extracted poses.
    # Because `preprocess` is now fixed, `pose_pixels` will have the correct length (e.g., 10 frames + 1 ref image).
    frames = pipeline(
        image_pixels, 
        image_pose=pose_pixels, 
        num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, 
        tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], 
        width=pose_pixels.shape[-1], 
        fps=7,
        noise_aug_strength=task_config.noise_aug_strength, 
        num_inference_steps=task_config.num_inference_steps,
        generator=generator, 
        min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, 
        decode_chunk_size=8, 
        output_type="pt", 
        device=device
    ).frames.cpu()
    
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # The first frame is deprecated because it is the reference image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(infer_config, device)

    for task in infer_config.test_case:
        ############################################## Pre-process data ##############################################
        logger.info(f"Processing video: {task.ref_video_path} with image: {task.ref_image_path}")
        logger.info(f"Requesting a final video of {task.num_frames} frames.")
        
        # Pass `task.num_frames` from the config into the preprocess function
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path, 
            task.ref_image_path, 
            resolution=task.resolution, 
            sample_stride=task.sample_stride,
            num_frames=task.num_frames
        )
        
        ########################################### Run MimicMotion pipeline ###########################################
        logger.info(f"Running MimicMotion pipeline...")
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, 
            pose_pixels, 
            device, 
            task
        )
        
        ################################### Save results to output folder ###########################################
        output_path = f"{args.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        save_to_mp4(
            _video_frames, 
            output_path,
            fps=task.fps,
        )
        logger.info(f"Video saved to {output_path}")

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml", help="Path to the inference config file.")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="Directory to save the output videos.")
    parser.add_argument("--log_file", type=str, default=None, help="Path to save the log file.")
    parser.add_argument("--no_use_float16", action="store_true", help="Disable float16 for inference.")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    log_filename = args.log_file if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    set_logger(log_filename)
    
    main(args)
    logger.info(f"--- Finished ---")