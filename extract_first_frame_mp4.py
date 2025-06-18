import cv2
import os

# ---- Configuration: Set these paths ----
video_path = r'C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\videos\sample_hb_1.mp4'        # Replace with your MP4 file path
output_folder = r'C:\Users\jaira\Desktop\AnnaOS\Apps\MimicMotion\assets\example_data\images'      # Replace with the folder to save JPG
output_filename = 'frame1.jpg'          # Optional: Customize output filename

# ---- Ensure output folder exists ----
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, output_filename)

# ---- Read the video and extract the first frame ----
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if success:
    cv2.imwrite(output_path, frame)
    print(f"First frame saved to: {output_path}")
else:
    print("Failed to read the video or extract the first frame.")
