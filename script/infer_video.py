from ultralytics import YOLO
import cv2
import time

# Nutze ein kompatibles, bereits verwendetes Modell (train8)
model = YOLO("runs/detect/train8/weights/best.pt")

# Eingangsvideo aus deinem video/-Ordner
video_path = "video/real_seq_000_part_0_dif_1.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = "video/12output_real_seq_000_part_0_dif_1.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.1, imgsz=640)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.time()
cap.release()
out.release()
cv2.destroyAllWindows()

elapsed_time = end_time - start_time
video_fps = frame_count / elapsed_time
print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds.")
print(f"Average video inference speed: {video_fps:.2f} FPS")
