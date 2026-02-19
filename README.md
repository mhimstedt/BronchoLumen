<div id="top" align="center">

# 🫁 BronchoLumen: Real-time bronchial orifice detection in video-bronchoscopy
  Yongchao Li, Marian Himstedt

  [![Journal](https://img.shields.io/badge/IJCARS-0000.00000-00000.svg)]()
  [![arXiv](https://img.shields.io/badge/arXiv-0000.00000-00000.svg)]()
  [![Download Model](https://img.shields.io/badge/Model-BronchoLumen-blue.svg)]()


</div>

## Overview

This repository contains a YOLO‑based project for detecting bronchial orifices in bronchoscopy images and videos.  
It is set up for reproducible training, evaluation, and inference using [`uv`](https://github.com/astral-sh/uv).

---

## 🗂 Project Structure

**Key directories & files**

- `data.yaml` – central YOLO configuration for data paths and classes
- `datasets/`
  - `train/images`, `train/labels` – training data (YOLO format)
  - `val/images`, `val/labels` – validation data
  - `test1/images`, `test1/labels` – test set 1 (used by `predict_yolo.py`)
  - `test2/images`, `test2/labels` – test set 2 (used as `split="test"`)
  - `val/vis/` – visualized ground‑truth labels
- `script/`
  - `train_yolo.py` – train the YOLO model
  - `test_yolo.py` – evaluate the model on the test split (`split="test"`)
  - `predict_yolo.py` – run inference on test images and save results
  - `infer_video.py` – run video inference with live visualization and output video
  - helper scripts (`polygon_to_yolo.py`, `convert_yolo_labels.py`, …)
- `runs/`
  - training and inference outputs (`runs/detect/...`)
- `video/`
  - bronchoscopy videos used for inference (`*.mp4`)
- `yolov8m.pt` – base checkpoint used for training

---

## 🛠️ Environment Setup (with `uv`)

### 1. Prerequisites

- Python ≥ **3.12**
- [`uv`](https://github.com/astral-sh/uv) installed, e.g.:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash

# Install dependencies from pyproject.toml
uv sync
```

`uv` will automatically create a virtual environment (e.g. `.venv/`) and install:

- `ultralytics` (YOLOv8)

### 3. Activate the virtual environment (optional)

In most cases `uv run` is enough, but you can also activate the venv explicitly:

```bash
source .venv/bin/activate   # macOS/Linux
```

---

## 📁 Data Layout & Paths

The file `data.yaml` is already configured for this project:

```5:8:data.yaml
train: datasets/train/images
val:   datasets/val/images
test:  datasets/test2/images

nc: 1
names:
  - orifice
```

**Important**

- For every image in `datasets/*/images` there must be a corresponding `.txt` file in `datasets/*/labels` (YOLO format: `class x_center y_center width height` in relative coordinates).
- The class `orifice` has **ID 0**.

If you move the datasets to a different location, only update the `train`, `val`, and `test` paths in `data.yaml`.

---

## 🚀 Training a YOLO Model

### Training script

The main training script is `script/train_yolo.py`:

```1:37:script/train_yolo.py
from ultralytics import YOLO
import torch


def get_device() -> str | int:
    """
    Select a suitable device automatically:
    - Apple Silicon: 'mps'
    - CUDA GPU: 0
    - otherwise CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main() -> None:
    """
    Trains a YOLO model using the datasets defined in data.yaml.
    """
    device = get_device()

    # Base checkpoint (standard YOLOv8m without A2 custom modules)
    model = YOLO("yolov8m.pt")

    model.train(
        data="data.yaml",  # uses the relative paths from data.yaml
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        workers=4,
        augment=True,
        name="yolov8m-retrain",
        pretrained=False,
        seed=42,
    )


if __name__ == "__main__":
    main()
```

### Start training

```bash
cd broncholumen
uv run python script/train_yolo.py
```

Results are stored in:

- `runs/detect/yolov8m-retrain/`
  - `weights/best.pt`
  - `weights/last.pt`
  - training plots (`results.png`, confusion matrix, PR curves, …)

---

## 📊 Evaluation in Test Mode (`split="test"`)

To evaluate the model on the dedicated test split (`datasets/test2`), use `script/test_yolo.py`:

```1:16:script/test_yolo.py
from ultralytics import YOLO


def main() -> None:
    """
    Evaluates the trained YOLO model on the test split defined in data.yaml.
    """
    # Use an existing trained model – currently train8
    model = YOLO("runs/detect/train8/weights/best.pt")

    # Validation in test mode (split="test"), paths come from data.yaml
    metrics = model.val(data="data.yaml", split="test")

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()
```

> 💡 After retraining, you can switch to the latest model, for example:  
> `model = YOLO("runs/detect/yolov8m-retrain/weights/best.pt")`.

### Run evaluation

```bash
uv run python script/test_yolo.py
```

mAP and other metrics are printed to the terminal and also logged under `runs/detect/...`.

---

## 🖼️ Batch Inference on Test Images

For batch prediction on a test set, use `script/predict_yolo.py`:

```1:27:script/predict_yolo.py
from ultralytics import YOLO

def main():
    # Use an already trained model (train8)
    model = YOLO("runs/detect/train8/weights/best.pt")

    # Run prediction on test set with adjusted NMS settings
    results = model.predict(
        source="datasets/test1/images",
        data="data.yaml",
        split="test",
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/detect",
        name="predict_test",
        imgsz=640,
        conf=0.1,
        iou=0.4,
        max_det=20,
        agnostic_nms=True
    )

    print("Prediction on test set complete.")
    print(results)

if __name__ == "__main__":
    main()
```

### Run prediction

```bash
uv run python script/predict_yolo.py
```

Outputs are stored in:

- `runs/detect/predict_test/`
  - annotated images (`*.jpg`)
  - YOLO prediction files (`labels/*.txt` with confidences)

You can adjust `source`, `conf`, `iou`, `name`, etc. in the script to experiment with other datasets and thresholds.

---

## 🎥 Video Inference with Live Visualization

For video inference (live window plus output video), use `script/infer_video.py`:

```1:32:script/infer_video.py
from ultralytics import YOLO
import cv2
import time

# Use a compatible, already trained model (train8)
model = YOLO("runs/detect/train8/weights/best.pt")

# Input video from the video/ directory
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
```

### Run video inference

```bash
uv run python script/infer_video.py
```

- A window titled **"YOLOv8 Detection"** shows the annotated video stream.
- Press **q** to stop playback.
- The processed video is written to `video/12output_real_seq_000_part_0_dif_1.mp4`.

> 🎛️ To use a different video, change `video_path` and optionally `output_path`.

---

## 🧪 Visualizing Ground‑Truth Labels

To quickly inspect labels in the validation set, use `datasets/val/visualize_yolo_val.py`:

```1:45:datasets/val/visualize_yolo_val.py
import os
import cv2

# === Configuration ===
val_dir = os.path.join("datasets", "val")
image_dir = os.path.join(val_dir, "images")
label_dir = os.path.join(val_dir, "labels")
class_names = ["orifice"]  # single class according to data.yaml
show_max = None  # Maximum number of images to visualize (None = all)
save_dir = os.path.join(val_dir, "vis")  # Directory to save visualizations
os.makedirs(save_dir, exist_ok=True)

# === Iterate over images ===
for idx, img_file in enumerate(os.listdir(image_dir)):
    if show_max and idx >= show_max:
        break
    if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + ".txt")
    img = cv2.imread(image_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # Draw YOLO boxes
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls, x, y, bw, bh = map(float, parts)
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = class_names[int(cls)] if int(cls) < len(class_names) else f"class_{int(cls)}"
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save visualization
    out_path = os.path.join(save_dir, img_file)
    cv2.imwrite(out_path, img)
    print(f"Saved: {out_path}")
```

### Run visualization

```bash
uv run python datasets/val/visualize_yolo_val.py
```

The visualized images are written to `datasets/val/vis/` and can be viewed in any image viewer or served via a simple HTTP server (`python -m http.server`).

---

## 🧩 Notes on Custom Architectures (A2 Backbone)

The file `yolov12m_a2.yaml` defines a custom backbone that uses an `A2` block:

```1:10:yolov12m_a2.yaml
# YOLOv12-M with A2 Attention Only (R_ELAN removed)

nc: 1  # number of classes
depth_multiple: 0.67
width_multiple: 0.75

backbone:
  - [-1, 1, Conv, [64, 3, 2]]  # 0
  - [-1, 1, Conv, [128, 3, 2]]  # 1
  - [-1, 1, A2, [128]]  # 2
```

Note: You have to make sure that your installed `ultralytics` version supports this custom architecture. 

---

## ✅ Command Cheat Sheet

- **Install environment & dependencies**

```bash
cd broncholumen
uv sync
```

- **Train**

```bash
uv run python script/train_yolo.py
```

- **Evaluate in test mode**

```bash
uv run python script/test_yolo.py
```

- **Batch prediction on test images**

```bash
uv run python script/predict_yolo.py
```

- **Video inference (live + output video)**

```bash
uv run python script/infer_video.py
```

- **Visualize ground‑truth labels**

```bash
uv run python datasets/val/visualize_yolo_val.py
```

Happy training and analysis! 🧪📊🫁

