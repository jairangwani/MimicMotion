#!/usr/bin/env python3
import os
import json
import numpy as np
import shutil
from copy import deepcopy
import cv2

# ==============================================================================
# === POSE VISUALIZATION HELPERS ===
# ==============================================================================
POSE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), 
    (9, 10), (1, 11), (11, 12), (12, 13), (0, 14), (14, 16), (0, 15), (15, 17)
]
TORSO_IDX, ARMS_IDX, LEGS_IDX, NOSE_IDX = {1, 2, 5, 8, 11}, {2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13}, {0}
EYE_IDX, LIP_IDX = set(range(36, 48)), set(range(48, 60))
COLOR_TORSO, COLOR_ARMS, COLOR_LEGS, COLOR_NOSE, COLOR_HAND, COLOR_FACE, COLOR_EYES, COLOR_LIPS = \
    (0, 128, 255), (255, 0, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (128, 128, 128), (255, 0, 128), (0, 0, 255)

def get_body_color(idx):
    """Gets the appropriate color for a given body keypoint index."""
    if idx in ARMS_IDX: return COLOR_ARMS
    if idx in LEGS_IDX: return COLOR_LEGS
    if idx in TORSO_IDX: return COLOR_TORSO
    if idx in NOSE_IDX: return COLOR_NOSE
    return COLOR_TORSO

def draw_pose_on_canvas(canvas: np.ndarray, pose: dict) -> np.ndarray:
    """Draws the body, face, and hand skeletons onto a canvas."""
    h, w = canvas.shape[:2]
    img = canvas.copy()
    if not pose: return img

    body_data = pose.get('bodies', {})
    body_candidate = body_data.get('candidate', [])
    body_subset = body_data.get('subset', [])
    
    if len(body_candidate) == 0 or len(body_subset) == 0: return img
    
    valid_keypoints = body_subset[0]
    for i, j in POSE_PAIRS:
        are_valid = i < len(valid_keypoints) and j < len(valid_keypoints) and valid_keypoints[i] > -1.0 and valid_keypoints[j] > -1.0
        if are_valid:
            idx1, idx2 = int(valid_keypoints[i]), int(valid_keypoints[j])
            if idx1 < len(body_candidate) and idx2 < len(body_candidate):
                (x1, y1), (x2, y2) = body_candidate[idx1], body_candidate[idx2]
                if i in ARMS_IDX and j in ARMS_IDX: line_color = COLOR_ARMS
                elif i in LEGS_IDX and j in LEGS_IDX: line_color = COLOR_LEGS
                else: line_color = COLOR_TORSO
                cv2.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), line_color, 2)
    
    for hand in pose.get('hands', []):
        for x, y in hand:
            if x > 0 and y > 0: cv2.circle(img, (int(x * w), int(y * h)), 3, COLOR_HAND, -1)
            
    for face in pose.get('faces', []):
        for f_idx, (x, y) in enumerate(face):
            if x > 0 and y > 0:
                c = COLOR_EYES if f_idx in EYE_IDX else COLOR_LIPS if f_idx in LIP_IDX else COLOR_FACE
                cv2.circle(img, (int(x * w), int(y * h)), 2, c, -1)
                
    return img

# ==============================================================================
# === POSE ARITHMETIC HELPERS ===
# ==============================================================================

def operate_on_poses(pose_a, pose_b, operation):
    """A generic helper to add or subtract pose data. Operation is np.add or np.subtract."""
    if pose_a is None or pose_b is None: return None
    result = deepcopy(pose_a)
    try:
        if 'bodies' in result and 'bodies' in pose_b and result['bodies'].get('candidate') is not None and pose_b['bodies'].get('candidate') is not None:
             result['bodies']['candidate'] = operation(np.array(result['bodies']['candidate']), np.array(pose_b['bodies']['candidate']))
        if 'faces' in result and 'faces' in pose_b and result.get('faces') is not None and pose_b.get('faces') is not None:
             result['faces'] = operation(np.array(result['faces']), np.array(pose_b['faces']))
        if 'hands' in result and 'hands' in pose_b and result.get('hands') is not None and pose_b.get('hands') is not None:
            for i in range(len(result['hands'])):
                result['hands'][i] = operation(np.array(result['hands'][i]), np.array(pose_b['hands'][i]))
    except (ValueError, IndexError) as e:
        print(f"Warning: Mismatched keypoints during pose operation. {e}")
    return result

def multiply_pose_by_scalar(pose, scalar):
    """Multiplies all keypoint coordinates in a pose by a scalar value."""
    if pose is None: return None
    result = deepcopy(pose)
    if 'bodies' in result and result['bodies'].get('candidate') is not None: result['bodies']['candidate'] = np.array(result['bodies']['candidate']) * scalar
    if 'faces' in result and result.get('faces') is not None: result['faces'] = np.array(result['faces']) * scalar
    if 'hands' in result and result.get('hands'):
        for i in range(len(result['hands'])):
            result['hands'][i] = np.array(result['hands'][i]) * scalar
    return result

# ==============================================================================
# === CORE ANIMATION RETARGETING LOGIC ===
# ==============================================================================

class NpEncoder(json.JSONEncoder):
    """Helper class to convert numpy types for JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)): return obj.item()
        return super().default(obj)

def load_pose_from_json(file_path: str) -> dict:
    """Loads a pose JSON and converts coordinates to numpy arrays."""
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}"); return None
    with open(file_path, 'r') as f:
        try: data = json.load(f)
        except json.JSONDecodeError: print(f"Error: Could not decode JSON from {file_path}"); return None
    if 'bodies' in data and data['bodies'].get('candidate'): data['bodies']['candidate'] = np.array(data['bodies']['candidate'])
    if 'faces' in data and data.get('faces'): data['faces'] = np.array(data['faces'])
    if 'hands' in data and data.get('hands'): data['hands'] = [np.array(hand) for hand in data['hands']]
    return data

def retarget_animation_with_anchors(source_dir: str, dest_start_frame_path: str, dest_end_frame_path: str, new_animation_name: str, canvas_size: tuple):
    """Retargets an animation to fit between a new start and end pose."""
    output_dir = new_animation_name
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Removing it.")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

    dest_start_pose = load_pose_from_json(dest_start_frame_path)
    dest_end_pose = load_pose_from_json(dest_end_frame_path)
    source_json_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.json')])
    if len(source_json_files) < 2:
        print("Error: Source directory must contain at least two frames."); return
    source_start_pose = load_pose_from_json(os.path.join(source_dir, source_json_files[0]))
    source_end_pose = load_pose_from_json(os.path.join(source_dir, source_json_files[-1]))
    if not all([dest_start_pose, dest_end_pose, source_start_pose, source_end_pose]):
        print("Error loading one or more required anchor frames. Aborting."); return

    total_source_travel = operate_on_poses(source_end_pose, source_start_pose, np.subtract)
    total_dest_travel = operate_on_poses(dest_end_pose, dest_start_pose, np.subtract)
    final_gap = operate_on_poses(total_dest_travel, total_source_travel, np.subtract)
    
    print(f"Blending {len(source_json_files)} frames to fit between new start and end poses...")
    canvas_template = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
    num_frames = len(source_json_files)

    for i, frame_filename in enumerate(source_json_files):
        current_source_pose = load_pose_from_json(os.path.join(source_dir, frame_filename))
        progress = i / (num_frames - 1) if num_frames > 1 else 0
        
        source_motion = operate_on_poses(current_source_pose, source_start_pose, np.subtract)
        new_pose_base = operate_on_poses(dest_start_pose, source_motion, np.add)
        correction_for_frame = multiply_pose_by_scalar(final_gap, progress)
        final_new_pose = operate_on_poses(new_pose_base, correction_for_frame, np.add)

        if final_new_pose and 'bodies' in final_new_pose and 'bodies' in current_source_pose and current_source_pose['bodies'].get('subset') is not None:
            final_new_pose['bodies']['subset'] = current_source_pose['bodies']['subset']

        json_filename = f"frame_{i:06d}.json"
        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(final_new_pose, f, indent=4, cls=NpEncoder)

        viz_image = draw_pose_on_canvas(canvas_template, final_new_pose)
        viz_filename = f"viz_{i:06d}.png"
        cv2.imwrite(os.path.join(output_dir, viz_filename), viz_image)
        
        if (i+1) % 20 == 0 or i == num_frames - 1:
            print(f"  Processed and saved: {json_filename} & {viz_filename}")

    print(f"\n✅ Processing complete. New blended animation saved in: '{output_dir}'")

# ==============================================================================
# === SCRIPT CONFIGURATION ===
# ==============================================================================
if __name__ == '__main__':
    # 1. Path to the source animation's JSON folder (provides the motion).
    SOURCE_DIRECTORY = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\videos\short_2_pose_json"

    # 2. Path to the JSON file for the DESIRED STARTING pose (first anchor frame).
    DESTINATION_START_FRAME = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\videos\short_2_pose_json\frame_000017.json"

    # 3. Path to the JSON file for the DESIRED ENDING pose (second anchor frame).
    DESTINATION_END_FRAME = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\videos\short_2_pose_json\frame_000017.json"

    # 4. Name for the new output folder.
    NEW_ANIMATION_NAME = "MyBlendedAnimation"
    
    # 5. Canvas size for the output images (Height, Width).
    CANVAS_SIZE = (1024, 576)

    # --- Run Script ---
    retarget_animation_with_anchors(
        source_dir=SOURCE_DIRECTORY,
        dest_start_frame_path=DESTINATION_START_FRAME,
        dest_end_frame_path=DESTINATION_END_FRAME,
        new_animation_name=NEW_ANIMATION_NAME,
        canvas_size=CANVAS_SIZE
    )