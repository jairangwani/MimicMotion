# inference.py  – full, fixed
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

# ------------------------------------------------------------------------
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
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose
# ------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------
def preprocess(
    video_path: str,
    image_path: str,
    *,
    resolution: int = 576,
    sample_stride: int = 2,
    num_frames: int | None = None,
    use_dw_from_path: bool = False,
    dw_data_path: str | None = None,
):
    """
    Returns
    -------
    pose_pixels : torch.Tensor (T+1, H, W, 3)  in [-1, 1], dtype = default
    image_pixels: torch.Tensor (1,   3, H, W)  in [-1, 1], dtype = default
    """
    img = pil_loader(image_path)
    img = pil_to_tensor(img)                     # (C,H,W) uint8
    h0, w0 = img.shape[-2:]

    if h0 > w0:
        w_t, h_t = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_t, h_t = int(resolution / ASPECT_RATIO // 64) * 64, resolution

    ratio = h0 / w0
    if ratio < h_t / w_t:
        h_r, w_r = h_t, math.ceil(h_t / ratio)
    else:
        h_r, w_r = math.ceil(w_t * ratio), w_t

    img = resize(img, [h_r, w_r], antialias=None)
    img = center_crop(img, [h_t, w_t])
    img_hwc = img.permute(1, 2, 0).numpy()

    # ---- poses ----------------------------------------------------------
    image_pose = get_image_pose(img_hwc)
    video_pose = get_video_pose(
        video_path,
        img_hwc,
        sample_stride=sample_stride,
        num_frames=num_frames,
        use_dw_from_path=use_dw_from_path,
        dw_data_path=dw_data_path,
    )
    logger.info(f"Successfully obtained {video_pose.shape[0]} pose frames")

    pose_pixels = np.concatenate([image_pose[None], video_pose])
    pose_pixels = (
        torch.from_numpy(pose_pixels.copy())
        .to(dtype=torch.get_default_dtype()) / 127.5 - 1.0
    )
    image_pixels = (
        img.unsqueeze(0)
        .to(dtype=torch.get_default_dtype()) / 127.5 - 1.0
    )
    return pose_pixels, image_pixels


@torch.no_grad()
def run_pipeline(
    pipeline: MimicMotionPipeline,
    image_pixels: torch.Tensor,
    pose_pixels: torch.Tensor,
    task_cfg,
):
    # Convert ref image back to PIL
    ref_uint8 = (
        ((image_pixels[0].to(torch.float32) + 1.0) * 127.5)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
    )
    pil_list = [to_pil_image(ref_uint8)]

    gen = torch.Generator(device=image_pixels.device)
    gen.manual_seed(task_cfg.seed)

    frames = (
        pipeline(
            pil_list,
            image_pose=pose_pixels,
            num_frames=pose_pixels.size(0),
            tile_size=task_cfg.num_frames,
            tile_overlap=task_cfg.frames_overlap,
            height=pose_pixels.shape[-2],
            width=pose_pixels.shape[-1],
            fps=task_cfg.fps,
            noise_aug_strength=task_cfg.noise_aug_strength,
            num_inference_steps=task_cfg.num_inference_steps,
            generator=gen,
            min_guidance_scale=task_cfg.guidance_scale,
            max_guidance_scale=task_cfg.guidance_scale,
            decode_chunk_size=8,
            output_type="pt",
            device=device,
        )
        .frames.cpu()
    )

    video_frames = (frames * 255).to(torch.uint8)[:, 1:]  # drop ref frame
    return video_frames[0]


# ------------------------------------------------------------------------
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

    for task in cfg.test_case:
        logger.info(f"Processing {task.ref_video_path}")
        pose, img = preprocess(
            task.ref_video_path,
            task.ref_image_path,
            resolution=task.resolution,
            sample_stride=task.sample_stride,
            num_frames=task.num_frames,
            use_dw_from_path=getattr(task, "use_dw_from_path", False),
            dw_data_path=getattr(task, "dw_data_path", None),
        )
        logger.info("Running MimicMotion pipeline…")
        vid = run_pipeline(pipeline, img, pose, task)

        out = (
            Path(args.output_dir)
            / f"{Path(task.ref_video_path).stem}_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        )
        save_to_mp4(vid, str(out), fps=task.fps)
        logger.info(f"Saved ⇒ {out}")

    logger.info("--- Finished ---")


if __name__ == "__main__":
    main_cli()
