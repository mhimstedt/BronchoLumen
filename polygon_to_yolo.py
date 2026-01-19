import os
import json
from shapely.geometry import Polygon

# Output directories
label_output_dir = "datasets/labels"
image_output_dir = "datasets/images"
os.makedirs(label_output_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)

# Image size (fixed 480x480)
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 480

# Mapping from label_id to YOLO class ID (0–8)
id_to_class = {
    "ce08e883-ddd2-4639-aa05-9fdac022f545": 0,  # Main carina
    "7e3b10f1-c8d2-4bf3-8250-1dc5954f2de5": 1,
    "c93010c7-b77b-44a8-8866-7511f989e97a": 2,
    "a9d13324-3c88-4135-88bf-7fbb4ccb4e13": 3,
    "1578d8da-1512-479c-aed7-ef7cd4fb5541": 4,
    "c115009a-d19a-4337-9151-0dd20a2562e7": 5,
    "e572ea16-52e8-4404-b866-363eb3f733ce": 6,
    "27cc98cd-7cad-4134-bfb5-b2c74af3326e": 7,
    "ccca7aa2-1593-4e16-a436-38a5516ce433": 8
}

# Load annotation file
with open("annotation.json", "r") as f:
    annotation_data = json.load(f)

# Build a map: object_id → image path
image_base_dir = "imgs"  # folder containing image subfolders
image_path_map = {}

for root, _, files in os.walk(image_base_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".png")):
            image_id = os.path.splitext(file)[0]
            image_path_map[image_id] = os.path.join(root, file)

# Convert annotations
for item in annotation_data:
    label_ids = item.get("label_ids", [])
    points = item.get("data", [])
    image_id = item.get("object_id")

    if not label_ids or not points or image_id not in image_path_map:
        continue

    for label_id in label_ids:
        if label_id not in id_to_class:
            continue

        class_id = id_to_class[label_id]
        polygon = Polygon([(pt["x"], pt["y"]) for pt in points])
        min_x, min_y, max_x, max_y = polygon.bounds

        # Convert to YOLO format 
        x_center = (min_x + max_x) / 2 / IMAGE_WIDTH
        y_center = (min_y + max_y) / 2 / IMAGE_HEIGHT
        width = (max_x - min_x) / IMAGE_WIDTH
        height = (max_y - min_y) / IMAGE_HEIGHT

        # Write label file
        txt_path = os.path.join(label_output_dir, f"{image_id}.txt")
        with open(txt_path, "a") as f:
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Copy image 
        dst_img_path = os.path.join(image_output_dir, f"{image_id}.jpg")
        if not os.path.exists(dst_img_path):
            from shutil import copyfile
            copyfile(image_path_map[image_id], dst_img_path)

print(" Polygon annotations converted to YOLO format successfully.")
