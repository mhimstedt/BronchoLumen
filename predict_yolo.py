from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train10/weights/best.pt")

    # Run prediction on test set with adjusted NMS settings
    results = model.predict(
        source="datasets/test/images",
        data="data.yaml",
        split="test",
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs/detect",
        name="predict_test",
        imgsz=640,
        conf=0.1,         # lower confidence threshold to include more boxes
        iou=0.4,          # lower NMS IoU threshold to reduce suppression
        max_det=20,       
        agnostic_nms=True 
    )

    print("Prediction on test set complete.")
    print(results)

if __name__ == "__main__":
    main()
