import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_keypoints_shape():
    # ——— Hardcoded paths — adjust as needed ———
    json_path    = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\outputs\pose1_keypoints.json"
    output_image = r"C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\outputs\first_frame_keypoints_shape.png"
    # ———————————————————————————————

    # 1) Load JSON
    with open(json_path, 'r') as f:
        all_kps = json.load(f)
    first_frame = all_kps[0]

    # 2) Extract and reshape coordinates
    # "bodies" is already a list of [x, y]
    bodies_px = np.array(first_frame["bodies"])    # shape (N_body, 2)

    # "faces" is nested as a single list of 68 points: [ [ [x,y], … ] ]
    faces_list = first_frame["faces"]               # length‐1 list
    if len(faces_list) != 1:
        raise RuntimeError("Unexpected nesting in 'faces'.")
    faces_px = np.array(faces_list[0])              # shape (N_face, 2)

    # "hands" is a list of two lists (left‐hand & right‐hand), each list of [x, y].
    # Stack them into one (N_hand_total, 2) array:
    hands_px = np.vstack(first_frame["hands"])      # shape (N_hand_total, 2)

    # 3) Compute tight bounding box around all points
    all_points = np.vstack([bodies_px, faces_px, hands_px])
    min_x, min_y = np.min(all_points, axis=0)
    max_x, max_y = np.max(all_points, axis=0)

    # 4) Add 5% margin
    x_margin = (max_x - min_x) * 0.05
    y_margin = (max_y - min_y) * 0.05
    x_limits = (min_x - x_margin, max_x + x_margin)
    # Invert Y‐axis so origin is at top
    y_limits = (max_y + y_margin, min_y - y_margin)

    # 5) Plot on blank canvas
    plt.figure(figsize=(6, 8))
    plt.scatter(bodies_px[:, 0], bodies_px[:, 1], c='red',   s=30, label='Body Joints',    marker='o')
    plt.scatter(faces_px[:, 0],  faces_px[:, 1],  c='lime',  s=10, label='Face Landmarks',  marker='x')
    plt.scatter(hands_px[:, 0],  hands_px[:, 1],  c='blue',  s=10, label='Hand Joints',     marker='^')

    plt.xlim(x_limits)
    plt.ylim(y_limits)
    plt.gca().set_aspect('equal', 'box')
    plt.axis('off')
    plt.legend(loc='upper right', fontsize='small')

    # 6) Save PNG
    Path(output_image).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"Saved keypoint shape plot to {output_image}")

if __name__ == "__main__":
    plot_keypoints_shape()
