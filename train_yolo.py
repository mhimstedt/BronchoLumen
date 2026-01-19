from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/yolov12m-4/weights/best.pt")
    
    model.train(
    data="C:/Users/Li553/yolo1/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device="cuda:0",
    workers=4,
    augment=True, 
    name="yolov12m-5",
    pretrained=False,
    seed=42

)
if __name__ == "__main__":
    main()