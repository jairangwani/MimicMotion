# test_video_write.py
import torch
from torchvision.io import write_video
import av

print("PyAV version:", av.__version__)

# Create a dummy video: 2 seconds of random RGB frames at 24 FPS
num_frames = 48
height, width = 64, 64
fps = 24

print("Generating dummy video frames...")
frames = (torch.rand(num_frames, height, width, 3) * 255).to(torch.uint8)

print("Saving to test_output.mp4...")
write_video("test_output.mp4", frames, fps=fps)

print("✅ Video successfully written as test_output.mp4")
