#!/usr/bin/env python3
"""
Batch script: read a series of pose JSON files and draw skeletons onto black canvases.
Different body parts and facial regions are colored distinctly.
Set the JSON directory, output directory, number of frames, and canvas size below.
"""
import os
import json
import cv2
import numpy as np

# COCO-style body pairs for drawing the skeleton
POSE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Nose → Neck → R shoulder → R elbow → R wrist
    (1, 5), (5, 6), (6, 7),              # Neck → L shoulder → L elbow → L wrist
    (1, 8), (8, 9), (9, 10),             # Neck → R hip → R knee → R ankle
    (1, 11), (11, 12), (12, 13),          # Neck → L hip → L knee → L ankle
    (0, 14), (14, 16),                    # Nose → R eye → R ear
    (0, 15), (15, 17)                     # Nose → L eye → L ear
]

# Body keypoint index groups
TORSO_IDX = {1, 2, 5, 8, 11}
ARMS_IDX  = {2, 3, 4, 5, 6, 7}
LEGS_IDX  = {8, 9, 10, 11, 12, 13}
NOSE_IDX  = {0}

# Facial landmark index groups (68-point scheme)
EYE_IDX   = set(range(36, 48))  # eyes
LIP_IDX   = set(range(48, 60))  # lips
# rest of face landmarks will be generic face color

# Colors (BGR)
COLOR_TORSO = (0, 128, 255)  # orange-ish
COLOR_ARMS  = (255, 0, 0)    # blue
COLOR_LEGS  = (0, 255, 255)  # yellow
COLOR_NOSE  = (0, 255, 0)    # green
COLOR_HAND  = (255, 0, 255)  # magenta
COLOR_FACE  = (128, 128, 128) # gray
COLOR_EYES  = (255, 0, 128)  # pink
COLOR_LIPS  = (0, 0, 255)    # red


def get_body_color(idx):
    if idx in ARMS_IDX:
        return COLOR_ARMS
    if idx in LEGS_IDX:
        return COLOR_LEGS
    if idx in TORSO_IDX:
        return COLOR_TORSO
    if idx in NOSE_IDX:
        return COLOR_NOSE
    return COLOR_TORSO


def draw_pose_on_canvas(canvas: np.ndarray, pose: dict) -> np.ndarray:
    """Draw body, face, and hand keypoints & skeleton on the black canvas."""
    h, w = canvas.shape[:2]
    img = canvas.copy()

    # --- Get body keypoints and validity information ---
    body_data = pose.get('bodies', {})
    body_candidate = body_data.get('candidate', [])
    body_subset = body_data.get('subset', [])

    # If no person is detected or subset is missing, return the canvas
    if not body_candidate or not body_subset:
        return img
        
    # Get the keypoint indices for the first detected person.
    # A value of -1.0 (or -1) means the keypoint was not detected.
    valid_keypoints = body_subset[0]

    # Draw body keypoints and skeleton
    # keypoints
    for idx, (x, y) in enumerate(body_candidate):
        # Only draw the keypoint if it was detected for this person.
        if idx < len(valid_keypoints) and valid_keypoints[idx] > -1.0:
            px, py = int(x * w), int(y * h)
            color = get_body_color(idx)
            cv2.circle(img, (px, py), 4, color, -1)

    # skeleton lines
    for i, j in POSE_PAIRS:
        # Check if both keypoints in the pair were detected.
        are_both_points_valid = (
            i < len(valid_keypoints) and j < len(valid_keypoints) and
            valid_keypoints[i] > -1.0 and valid_keypoints[j] > -1.0
        )
        
        if are_both_points_valid:
            # Get the actual indices from the candidate list
            idx1 = int(valid_keypoints[i])
            idx2 = int(valid_keypoints[j])

            # Ensure these indices are within the bounds of the candidate list
            if idx1 < len(body_candidate) and idx2 < len(body_candidate):
                (x1, y1) = body_candidate[idx1]
                (x2, y2) = body_candidate[idx2]
                
                # choose a color if both joints in same group, else generic torso
                if i in ARMS_IDX and j in ARMS_IDX:
                    line_color = COLOR_ARMS
                elif i in LEGS_IDX and j in LEGS_IDX:
                    line_color = COLOR_LEGS
                else:
                    line_color = COLOR_TORSO
                    
                cv2.line(img,
                         (int(x1 * w), int(y1 * h)),
                         (int(x2 * w), int(y2 * h)),
                         line_color, 2)

    # Draw hands
    for hand in pose.get('hands', []):
        for x, y in hand:
            px, py = int(x * w), int(y * h)
            if x > 0 and y > 0:
                cv2.circle(img, (px, py), 3, COLOR_HAND, -1)

    # Draw face landmarks
    for face in pose.get('faces', []):
        for f_idx, (x, y) in enumerate(face):
            px, py = int(x * w), int(y * h)
            if f_idx in EYE_IDX:
                c = COLOR_EYES
            elif f_idx in LIP_IDX:
                c = COLOR_LIPS
            else:
                c = COLOR_FACE
            if x > 0 and y > 0:
                cv2.circle(img, (px, py), 2, c, -1)

    return img


if __name__ == '__main__':
    # === Configuration: set these variables ===
    json_dir     = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\videos\short_2_pose_json"
    output_dir   = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\videos\short_2_pose_json"
    num_frames   = 200   # number of JSON frames to process

    ### FIXED ### Set canvas to match the original video resolution (Height, Width)
    canvas_size  = (1024,576)
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Prepare blank canvas
    canvas_template = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)

    print(f"Processing JSON files from: {json_dir}")
    print(f"Saving visualizations to: {output_dir}")
    print(f"Using canvas size (HxW): {canvas_size}")

    for idx in range(num_frames):
        filename = f"frame_{idx:06d}.json"
        json_path = os.path.join(json_dir, filename)
        
        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            try:
                pose_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for frame {idx:06d}, skipping.")
                continue

        viz = draw_pose_on_canvas(canvas_template, pose_data)

        out_name = f"viz_{idx:06d}.png"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, viz)
        
        if idx % 20 == 0:
            print(f"Saved skeleton for frame {idx:06d} → {out_path}")
            
    print("Processing complete.")


      