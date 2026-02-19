import os
import cv2

def detect_and_generate_yolo_labels(image_path, output_txt_path, class_id=0, min_area=200):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    with open(output_txt_path, 'w') as f:
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh < min_area:
                continue
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            norm_w = bw / w
            norm_h = bh / h
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def batch_generate_yolo_labels(image_folder, output_folder, extensions=(".bmp", ".jpg", ".png")):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(extensions):
            image_path = os.path.join(image_folder, filename)
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_folder, txt_filename)
            detect_and_generate_yolo_labels(image_path, txt_path)
            print(f"Processed: {filename}")


image_folder = "datasets/Groundtruth"
output_folder = "datasets/labels"

batch_generate_yolo_labels(image_folder, output_folder)
