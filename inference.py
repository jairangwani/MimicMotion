# inference.py - FINAL "TRUE DELTA" VERSION
import os
import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import (
    pil_to_tensor, resize, center_crop, to_pil_image,
)
import copy

from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

try:
    from constants import ASPECT_RATIO
except ImportError:
    ASPECT_RATIO = 576 / 1024
    print("[inference] constants.py missing – default ASPECT_RATIO used")

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose, draw_pose

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_true_delta_pose_sequence(base_kps, aligned_motion_kps):
    """
    Implements the "True Delta" logic:
    - The first frame's pose is the base character's pose.
    - Subsequent frames are `base_pose + delta`.
    - Dynamically appearing limbs are directly transferred.
    """
    if not aligned_motion_kps:
        return []

    final_kps_sequence = []
    start_motion_kps = aligned_motion_kps[0]

    for motion_frame_kps in aligned_motion_kps:
        # Start with the character's base pose as the foundation for every frame.
        new_pose_kps = copy.deepcopy(base_kps)

        # --- Body: Apply true delta to the base pose ---
        if "bodies" in base_kps and "bodies" in start_motion_kps and "bodies" in motion_frame_kps:
            body_delta = motion_frame_kps["bodies"]["candidate"] - start_motion_kps["bodies"]["candidate"]
            new_pose_kps["bodies"]["candidate"] = base_kps["bodies"]["candidate"] + body_delta
            # Update visibility from the motion frame
            new_pose_kps["bodies"]["subset"] = motion_frame_kps["bodies"]["subset"]
        
        # --- Faces & Hands: Use a hybrid delta/direct transfer logic ---
        for key in ["faces", "hands"]:
            if key in motion_frame_kps and len(motion_frame_kps.get(key, [])) > 0:
                current_parts = motion_frame_kps[key]
                start_parts = start_motion_kps.get(key, [])
                base_parts = base_kps.get(key, [])

                # Ensure the list for new parts exists
                if key not in new_pose_kps:
                    new_pose_kps[key] = []
                
                for i, current_part in enumerate(current_parts):
                     # Ensure the new list is long enough
                    while len(new_pose_kps[key]) <= i:
                        new_pose_kps[key].append(None)

                    base_part = base_parts[i] if i < len(base_parts) else None
                    start_part = start_parts[i] if i < len(start_parts) else None

                    # If the part exists on the base character, apply the delta to it.
                    if base_part is not None and start_part is not None:
                        part_delta = current_part - start_part
                        new_pose_kps[key][i] = base_part + part_delta
                    # Otherwise (part appears dynamically), perform a direct transfer.
                    else:
                        new_pose_kps[key][i] = current_part
        
        final_kps_sequence.append(new_pose_kps)
        
    return final_kps_sequence


def preprocess(task_cfg):
    """Prepares inputs using the final hybrid logic."""
    image_path = task_cfg.character_image_path
    video_path = task_cfg.motion_video_path
    
    img = pil_loader(image_path)
    
    # Letterboxing logic to preserve aspect ratio
    img_tensor = pil_to_tensor(img)
    _, h0, w0 = img_tensor.shape
    target_height = 576
    target_width = 1024
    scale = target_height / h0
    new_width = int(w0 * scale)
    new_height = target_height
    img_resized = resize(img_tensor, [new_height, new_width], antialias=True)
    canvas = torch.zeros((img_resized.shape[0], target_height, target_width), dtype=img_resized.dtype)
    pad_left = (target_width - new_width) // 2
    canvas[:, :, pad_left:pad_left + new_width] = img_resized
    final_processed_tensor = canvas
    
    img_hwc = final_processed_tensor.permute(1, 2, 0).numpy()

    # --- Get Base and Motion Data ---
    logger.info(f"Extracting base pose from: {Path(image_path).name}")
    character_pose_img, base_kps = get_image_pose(img_hwc)

    logger.info(f"Extracting and aligning motion sequence from video: {Path(video_path).name}")
    aligned_motion_kps = get_video_pose(
        video_path=video_path,
        ref_image=img_hwc,
        dw_pose_dir=Path(task_cfg.dw_pose_dir),
        sample_stride=task_cfg.sample_stride,
        num_frames=task_cfg.num_frames
    )
    logger.info(f"Obtained {len(aligned_motion_kps)} aligned motion keyframe sets.")

    # --- Create the "True Delta" Pose Sequence ---
    logger.info("Creating 'True Delta' pose sequence...")
    final_kps_sequence = create_true_delta_pose_sequence(base_kps, aligned_motion_kps)
    
    # --- Draw Final Pose Images ---
    h, w, _ = img_hwc.shape
    final_pose_images = [np.array(draw_pose(kps, h, w)) for kps in final_kps_sequence]
    
    if not final_pose_images:
        # Fallback for empty motion sequence
        all_pose_images_np = character_pose_img[None]
    else:
        # The first pose image is the one drawn from our True Delta logic,
        # which should perfectly match the character pose image.
        all_pose_images_np = np.stack(final_pose_images)
    
    # --- Convert to tensors for the pipeline ---
    image_pixels = (final_processed_tensor.unsqueeze(0).to(dtype=torch.get_default_dtype()) / 127.5 - 1.0)
    pose_pixels = (torch.from_numpy(all_pose_images_np.copy()).to(dtype=torch.get_default_dtype()) / 127.5 - 1.0)
    
    return pose_pixels, image_pixels


@torch.no_grad()
def run_pipeline(pipeline, image_pixels, pose_pixels, task_cfg):
    """Standard pipeline execution function."""
    ref_uint8 = (((image_pixels[0].to(torch.float32) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).cpu())
    pil_list = [to_pil_image(ref_uint8)]
    gen = torch.Generator(device=image_pixels.device)
    gen.manual_seed(task_cfg.seed)
    
    # The pipeline should receive a pose sequence of the same length as the number of frames to generate
    num_gen_frames = pose_pixels.size(0)

    frames = pipeline(
        pil_list, image_pose=pose_pixels, num_frames=num_gen_frames,
        tile_size=task_cfg.num_frames, tile_overlap=task_cfg.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=task_cfg.fps,
        noise_aug_strength=task_cfg.noise_aug_strength, num_inference_steps=task_cfg.num_inference_steps,
        generator=gen, min_guidance_scale=task_cfg.guidance_scale, max_guidance_scale=task_cfg.guidance_scale,
        decode_chunk_size=8, output_type="pt", device=device,
    ).frames.cpu()
    
    # The pipeline output includes the initial image, so we return all frames.
    video_frames = (frames * 255).to(torch.uint8)
    return video_frames[0]


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference_config", default="configs/test.yaml")
    ap.add_argument("--output_dir", default="outputs/")
    ap.add_argument("--log_file")
    ap.add_argument("--no_use_float16", action="store_true")
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    if args.log_file:
        fh = logging.FileHandler(args.log_file, "w")
        fh.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        logger.addHandler(fh)

    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    cfg = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(cfg, device)

    for i, task in enumerate(cfg.test_case):
        logger.info(f"--- Processing Task {i+1}/{len(cfg.test_case)} ---")
        pose, img = preprocess(task)
        logger.info("Running MimicMotion pipeline…")
        vid = run_pipeline(pipeline, img.to(device), pose.to(device), task)
        char_name = Path(task.character_image_path).stem
        motion_name = Path(task.motion_video_path).stem
        out_name = f"{char_name}_on_{motion_name}_truedelta_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        out_path = Path(args.output_dir) / out_name
        save_to_mp4(vid, str(out_path), fps=task.fps)
        logger.info(f"Saved ⇒ {out_path}")

    logger.info("--- Finished ---")


if __name__ == "__main__":
    main_cli()