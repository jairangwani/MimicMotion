import os
import json
from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


class NpEncoder(json.JSONEncoder):
    """Helper to convert numpy types to native Python for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)


def get_video_pose(
        video_path: str,
        ref_image: np.ndarray,
        sample_stride: int = 1,
        num_frames: int = None
    ) -> np.ndarray:
    """
    Preprocess reference image pose and extract pose + save JSON for each frame.

    Args:
        video_path (str): path to the input video
        ref_image (np.ndarray): reference image for pose rescale
        sample_stride (int): frame sampling stride
        num_frames (int): limit on number of frames (None => all)

    Returns:
        np.ndarray: sequence of pose-drawn frames (H x W x 3)
    """
    # --- derive directory for JSON output ---
    base = os.path.splitext(os.path.basename(video_path))[0]
    json_dir = os.path.join(os.path.dirname(video_path), f"{base}_pose_json")
    os.makedirs(json_dir, exist_ok=True)

    # Reference keypoints
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    if len(ref_pose['bodies']['subset']) > 0:
        ref_keypoint_id = [i for i in ref_keypoint_id
                           if ref_pose['bodies']['subset'][0][i] >= 0.0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # Video reading & frame sampling
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))
    all_indices = list(range(0, len(vr), sample_stride))
    if num_frames is not None:
        all_indices = all_indices[:num_frames]
    frames = vr.get_batch(all_indices).asnumpy()

    # Pose detection
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    # Stack bodies for rescale
    detected_bodies = np.stack([
        p['bodies']['candidate'] for p in detected_poses
        if p['bodies']['candidate'].shape[0] == 18
    ])[:, ref_keypoint_id]

    # Linear rescale parameters
    ay, by = np.polyfit(
        detected_bodies[:, :, 1].flatten(),
        np.tile(ref_body[:, 1], len(detected_bodies)), 1
    )
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(
        np.tile(ref_body[:, 0], len(detected_bodies))
        - detected_bodies[:, :, 0].flatten() * ax
    )
    a = np.array([ax, ay])
    b = np.array([bx, by])

    output_pose = []
    # Process each detected pose: rescale, save JSON, draw
    for idx, detected_pose in enumerate(detected_poses):
        # Rescale all parts
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b

        # --- Save pose data as JSON ---
        json_path = os.path.join(json_dir, f"frame_{idx:06d}.json")
        with open(json_path, 'w') as jf:
            # dump full pose dict (numpy -> lists)
            json.dump(detected_pose, jf, indent=4, cls= NpEncoder)

        # Draw and store pose image
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))

    return np.stack(output_pose)


def get_image_pose(ref_image: np.ndarray) -> np.ndarray:
    """
    Process single image pose.

    Args:
        ref_image (np.ndarray): reference image pixel data

    Returns:
        np.ndarray: pose visualization (RGB)
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)
