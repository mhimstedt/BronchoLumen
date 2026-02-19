from ultralytics import YOLO


def main() -> None:
    """
    Evaluates the trained YOLO model on the test split defined in data.yaml.
    """
    # Verwende ein existierendes Modell – hier train8
    model = YOLO("runs/detect/train8/weights/best.pt")

    # Validierung im Testmodus (split="test"), Pfade kommen aus data.yaml
    metrics = model.val(data="data.yaml", split="test")

    print("Evaluation complete.")
    print(metrics)


if __name__ == "__main__":
    main()