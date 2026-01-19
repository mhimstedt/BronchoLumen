from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/train8/weights/best.pt")
    metrics = model.val(data="data.yaml", split="test")  

    print(" Evaluation complete.")
    print(metrics)

if __name__ == "__main__":
    main()