
import cv2
import os

# === Configuration ===
video_path = "output_real_seq_000_part_0_dif_1.mp4"  
output_dir = "frames_output"   
os.makedirs(output_dir, exist_ok=True)

# === Open the video ===
cap = cv2.VideoCapture(video_path)
frame_count = 0

if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# === Read and save frames ===
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_name = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_name, frame)
    frame_count += 1

cap.release()
print(f"Saved {frame_count} frames to '{output_dir}'")
