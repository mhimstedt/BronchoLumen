from ultralytics import YOLO
import torch


def get_device() -> str | int:
    """
    Wählt automatisch ein sinnvolles Device:
    - M‑Chip: 'mps'
    - CUDA-GPU: 0
    - sonst CPU
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main() -> None:
    """
    Trainiert ein YOLO-Modell mit den in data.yaml definierten Datensätzen.
    """
    device = get_device()

    # Basis‑Checkpoint (Standard-YOLOv8m ohne A2-Sondermodul)
    model = YOLO("yolov8m.pt")

    model.train(
        data="data.yaml",  # nutzt die relativen Pfade in data.yaml
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