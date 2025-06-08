from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray, 
        sample_stride: int=1,
        num_frames: int=None): # MODIFIED: Added num_frames parameter
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.
        num_frames (int, optional): The number of frames to extract. 
                                    If None, processes the entire video. Defaults to None.

    Returns:
        np.ndarray: sequence of video pose
    """
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))

    # MODIFIED: The following block limits the number of frames to be processed
    # 1. Generate indices for all frames based on the sample_stride
    all_indices = list(range(0, len(vr), sample_stride))
    
    # 2. If num_frames is specified, truncate the list of indices
    if num_frames is not None:
        all_indices = all_indices[:num_frames]
    
    # 3. Load only the selected frames from the video
    frames = vr.get_batch(all_indices).asnumpy()
    # END OF MODIFICATION

    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()
    
    # The rest of the function remains unchanged. 
    # It will now correctly operate on the limited number of frames.

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                                        ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)