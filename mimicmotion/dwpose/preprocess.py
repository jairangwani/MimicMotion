# mimicmotion/dwpose/preprocess.py - DEFINITIVE FINAL ValueError FIX
import os
import json
import re
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple

from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return obj.item()
        return super().default(obj)

def get_video_pose(
    video_path: str,
    ref_image: np.ndarray,
    dw_pose_dir: Union[str, Path],
    sample_stride: int = 1,
    num_frames: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Runs DWPose, performs skeleton alignment, and returns a list of ALIGNED raw keypoint dictionaries.
    """
    h, w, _ = ref_image.shape
    json_dir = Path(dw_pose_dir) / f"{Path(video_path).stem}_poses"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))
    frame_idx = list(range(0, len(vr), sample_stride))
    if num_frames is not None:
        frame_idx = frame_idx[:num_frames]
    frames = vr.get_batch(frame_idx).asnumpy()

    all_kps = [dwprocessor(f) for f in tqdm(frames, desc=f"DWPose on {Path(video_path).name}")]
    dwprocessor.release_memory()

    ref_pose = dwprocessor(ref_image)
    ref_kpt_idx = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    if len(ref_pose["bodies"].get("subset", [])) > 0:
        ref_kpt_idx = [i for i in ref_kpt_idx if ref_pose["bodies"]["subset"][0][i] >= 0]
    
    if ref_kpt_idx and len(ref_pose["bodies"].get("candidate", [])) > 0:
        ref_body = ref_pose["bodies"]["candidate"][ref_kpt_idx]
        det_bodies = np.stack([
            p["bodies"]["candidate"] for p in all_kps if "candidate" in p.get("bodies", {}) and p["bodies"]["candidate"].shape[0] == 18
        ])
        if det_bodies.shape[0] > 0:
            det_bodies = det_bodies[:, ref_kpt_idx]
            ay, by = np.polyfit(det_bodies[:, :, 1].ravel(), np.tile(ref_body[:, 1], len(det_bodies)), 1)
            fh, fw, _ = vr[0].shape
            ax = ay / (fh / fw / h * w) if (fh / fw / h * w) != 0 else ay
            bx = np.mean(np.tile(ref_body[:, 0], len(det_bodies)) - det_bodies[:, :, 0].ravel() * ax)
            a = np.array([ax, ay])
            b = np.array([bx, by])

            for kps_dict in all_kps:
                if "candidate" in kps_dict.get("bodies",{}): kps_dict["bodies"]["candidate"] = kps_dict["bodies"]["candidate"] * a + b
                
                # --- CORRECTED ROBUST CHECKS ---
                if "faces" in kps_dict and len(kps_dict["faces"]) > 0: 
                    kps_dict["faces"] = kps_dict["faces"] * a + b
                if "hands" in kps_dict and len(kps_dict["hands"]) > 0: 
                    kps_dict["hands"] = kps_dict["hands"] * a + b
    
    for idx, kps_dict in enumerate(all_kps):
        jf = json_dir / f"frame_{idx:06d}.json"
        with open(jf, "w") as f:
            json.dump(kps_dict, f, cls=NpEncoder)
    
    return all_kps


def get_image_pose(ref_image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Runs DWPose and returns both the drawn image AND the raw keypoint data.
    """
    h, w, _ = ref_image.shape
    kps_dict = dwprocessor(ref_image)
    dwprocessor.release_memory()
    drawn_pose = np.array(draw_pose(kps_dict, h, w))
    return drawn_pose, kps_dict