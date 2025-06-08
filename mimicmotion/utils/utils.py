import logging
from pathlib import Path
import torch
import imageio # Use imageio instead of torchvision.io

logger = logging.getLogger(__name__)

def save_to_mp4(frames: torch.Tensor, save_path: str, fps: int = 7):
    """
    Saves a tensor of video frames to an MP4 file using the imageio library.

    Args:
        frames (torch.Tensor): A tensor of frames with shape (f, c, h, w)
                               and dtype torch.uint8.
        save_path (str): The path where the MP4 file will be saved.
        fps (int): The frames per second for the output video.
    """
    # Ensure the parent directory for the output file exists
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # imageio expects the frames in (f, h, w, c) format.
    # The tensor from the pipeline is (f, c, h, w), so we permute the dimensions.
    frames_for_saving = frames.permute(0, 2, 3, 1).cpu().numpy()

    # Write the video file using imageio
    # The 'quality' parameter is optional (0-10 scale), 8 is a good balance.
    imageio.mimwrite(save_path, frames_for_saving, fps=fps, quality=8)