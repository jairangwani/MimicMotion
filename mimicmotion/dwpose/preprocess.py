# mimicmotion/dwpose/preprocess.py
import os
import json
import re
from pathlib import Path
from typing import List, Union, Optional

from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


class NpEncoder(json.JSONEncoder):
    """Convert NumPy scalars/arrays to plain Python for JSON dumping."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Helper that converts every *nested* list inside a cached pose back to np.ndarray
# ---------------------------------------------------------------------------
def _lists_to_numpy(tree):
    if isinstance(tree, list):
        # recurse
        return np.asarray([_lists_to_numpy(item) for item in tree], dtype=np.float32)
    return tree


# ---------------------------------------------------------------------------
# Main helpers
# ---------------------------------------------------------------------------
def get_video_pose(
    video_path: str,
    ref_image: np.ndarray,
    sample_stride: int = 1,
    num_frames: Optional[int] = None,
    use_dw_from_path: bool = False,
    dw_data_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """
    Either run the DWPose detector (and save JSON) **or**
    read pre-computed JSON files and return the same pose-image tensor.

    Returns
    -------
    np.ndarray : shape (T, H, W, 3) with dtype uint8
    """
    # 1) Decide where JSON lives
    base           = Path(video_path).stem
    default_json   = Path(video_path).with_name(f"{base}_pose_json")
    json_dir       = Path(dw_data_path) if dw_data_path else default_json

    # ------------------------------------------------------------------
    # 2) ===== RE-USE cached pose ===================================================
    # ------------------------------------------------------------------
    if use_dw_from_path:
        if not json_dir.exists():
            raise FileNotFoundError(
                f"--use_dw_from_path was set but no JSON folder found at '{json_dir}'."
            )

        # Accept 'frame_000123.json' etc.  Robust to gaps/missing frames.
        json_files: List[Path] = sorted(
            f for f in json_dir.iterdir() if re.fullmatch(r"frame_\d{6}\.json", f.name)
        )
        if not json_files:
            raise RuntimeError(f"No frame_*.json files found in '{json_dir}'.")
        if num_frames is not None:
            json_files = json_files[:num_frames]

        h, w, _ = ref_image.shape
        pose_imgs = []
        for jf in json_files:
            with open(jf, "r") as f:
                cached_pose = json.load(f)

            # ---- convert every list back to NumPy (draw_pose needs that) ----
            cached_pose["bodies"]["candidate"] = np.asarray(
                cached_pose["bodies"]["candidate"], dtype=np.float32
            )
            cached_pose["bodies"]["subset"]    = np.asarray(
                cached_pose["bodies"]["subset"],    dtype=np.float32
            )
            cached_pose["bodies"]["score"]     = np.asarray(
                cached_pose["bodies"]["score"],     dtype=np.float32
            )

            # hands / faces blocks are lists-of-lists; keep that structure
            for key in ("hands", "hands_score", "faces", "faces_score"):
                if key in cached_pose and cached_pose[key]:
                    cached_pose[key] = [
                        np.asarray(item, dtype=np.float32) for item in cached_pose[key]
                    ]

            pose_imgs.append(np.array(draw_pose(cached_pose, h, w)))

        return np.stack(pose_imgs)   # (T, H, W, 3), dtype=uint8

    # ------------------------------------------------------------------
    # 3) ===== NEW detection & save JSON ==========================================
    # ------------------------------------------------------------------
    json_dir.mkdir(parents=True, exist_ok=True)

    # Reference keypoints for linear rescale
    ref_pose        = dwprocessor(ref_image)
    ref_kpt_idx     = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    if len(ref_pose["bodies"]["subset"]):
        ref_kpt_idx = [i for i in ref_kpt_idx if ref_pose["bodies"]["subset"][0][i] >= 0]
    ref_body        = ref_pose["bodies"]["candidate"][ref_kpt_idx]

    h, w, _ = ref_image.shape

    # ---------------- Video reading & frame sampling -------------------
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))
    frame_idx      = list(range(0, len(vr), sample_stride))
    if num_frames is not None:
        frame_idx = frame_idx[:num_frames]
    frames         = vr.get_batch(frame_idx).asnumpy()      # (T, H, W, 3)

    # ---------------- DWPose inference ---------------------------------
    detected = [dwprocessor(f) for f in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    # Linear rescale so that body size matches ref image
    det_bodies = np.stack([
        p["bodies"]["candidate"] for p in detected
        if p["bodies"]["candidate"].shape[0] == 18
    ])[:, ref_kpt_idx]

    ay, by = np.polyfit(det_bodies[:, :, 1].ravel(),
                        np.tile(ref_body[:, 1], len(det_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / h * w)
    bx = np.mean(
        np.tile(ref_body[:, 0], len(det_bodies)) -
        det_bodies[:, :, 0].ravel() * ax
    )
    a = np.array([ax, ay])
    b = np.array([bx, by])

    # ---------------- Save JSON & make pose images ---------------------
    pose_imgs = []
    for idx, det_pose in enumerate(detected):
        det_pose["bodies"]["candidate"] = det_pose["bodies"]["candidate"] * a + b
        det_pose["faces"]  = det_pose["faces"]  * a + b
        det_pose["hands"]  = det_pose["hands"]  * a + b

        # dump as JSON (lists!)
        jf = json_dir / f"frame_{idx:06d}.json"
        with open(jf, "w") as f:
            json.dump(det_pose, f, indent=4, cls=NpEncoder)

        pose_imgs.append(np.array(draw_pose(det_pose, h, w)))

    return np.stack(pose_imgs)   # (T, H, W, 3), dtype=uint8


def get_image_pose(ref_image: np.ndarray) -> np.ndarray:
    """Pose image for the *reference* still frame."""
    h, w, _ = ref_image.shape
    pose = dwprocessor(ref_image)
    return np.array(draw_pose(pose, h, w))
