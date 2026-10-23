"""Fine-tune a small YOLO on the auto-labelled ball crops (checkpoints go to Drive, so a dropped session can resume).

    python -m mtah.ft_train --data-yaml /content/ftdata/data.yaml --out /content/drive/MyDrive/mtah/ft
    python -m mtah.ft_train --out /content/drive/MyDrive/mtah/ft --resume          # after a disconnect
Best weights: <out>/ball/weights/best.pt
"""
import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    from ultralytics import YOLO
    last = os.path.join(a.out, "ball", "weights", "last.pt")
    if a.resume:
        YOLO(last).train(resume=True)
    else:
        YOLO(a.weights).train(data=a.data_yaml, epochs=a.epochs, imgsz=640, batch=a.batch, project=a.out, name="ball",
                              exist_ok=True, seed=0, deterministic=True, workers=2, mosaic=0.0, close_mosaic=0,
                              scale=0.25, patience=6, plots=False)
    print("[ft] best weights:", os.path.join(a.out, "ball", "weights", "best.pt"))


if __name__ == "__main__":
    main()
