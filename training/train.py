"""
train.py
========
YOLOv8-seg training script for document corner detection.

Trains a YOLOv8-nano segmentation model on a unified dataset assembled from
multiple document image collections and exports the best checkpoint to
TFLite for Android edge deployment.

Usage
-----
From the repository root::

    python training/train.py --data training/dataset.yaml

    # Resume an interrupted run
    python training/train.py --data training/dataset.yaml --resume

    # Custom hyperparameters
    python training/train.py \\
        --data training/dataset.yaml \\
        --model yolov8s-seg.pt \\
        --epochs 100 \\
        --batch 8 \\
        --imgsz 640 \\
        --device cpu

    # Export to TFLite after training (requires tensorflow ≥ 2.12)
    python training/train.py --data training/dataset.yaml --export

Notes
-----
* Trained weights are saved under ``runs/segment/<project>/<name>/weights/``.
* Copy ``weights/best.pt`` to the top-level ``weights/`` directory to use
  the scripts in ``src/``.
* See ``training/dataset.yaml`` for dataset path configuration.
"""

import argparse
import os
import shutil
import sys

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Resolve directory paths independent of the working directory
# ---------------------------------------------------------------------------
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_TRAIN_DIR)

DEFAULT_DATA = os.path.join(_TRAIN_DIR, "dataset.yaml")
DEFAULT_WEIGHTS_OUT = os.path.join(_ROOT_DIR, "weights")


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def run_training(
    data: str = DEFAULT_DATA,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 15,
    device: str = "0",
    project: str = "runs/segment/DocScan",
    name: str = "android_model",
    resume: bool = False,
    export: bool = False,
    workers: int = 8,
) -> str:
    """Train YOLOv8-seg on the document corner detection dataset.

    Parameters
    ----------
    data : str
        Path to the YOLO dataset YAML configuration file.
    model_name : str
        YOLOv8 model variant to use as the starting checkpoint.
        ``yolov8n-seg.pt`` (nano) is recommended for Android deployment.
        Ultralytics will download the pretrained weights automatically.
    epochs : int
        Maximum number of training epochs.
    imgsz : int
        Input image resolution (must match inference / export resolution).
    batch : int
        Mini-batch size.  Reduce to 8 or 4 if GPU VRAM is limited.
    patience : int
        Early stopping patience: halt training if validation metrics do not
        improve for this many consecutive epochs.
    device : str
        Training device.  ``"0"`` → first CUDA GPU; ``"cpu"`` → CPU only;
        ``"0,1"`` → multi-GPU.
    project : str
        Ultralytics output project directory (relative to cwd).
    name : str
        Run name within the project.  Incremented automatically if it
        already exists (e.g. ``android_model2``).
    resume : bool
        If True, resume the most recent interrupted training run.
    export : bool
        If True, export ``best.pt`` to FP16 and INT8 TFLite after training.
    workers : int
        Number of DataLoader worker processes.  Set to 0 on Windows if
        multiprocessing causes issues.

    Returns
    -------
    best_pt_path : str
        Absolute path to the best-performing checkpoint ``best.pt``.
    """
    if not os.path.isfile(data):
        raise FileNotFoundError(
            f"Dataset YAML not found: {data}\n"
            "Edit training/dataset.yaml to point to your assembled dataset."
        )

    print("=" * 60)
    print("🚀  YOLOv8-seg Document Corner Detection — Training")
    print("=" * 60)
    print(f"   Model   : {model_name}")
    print(f"   Data    : {data}")
    print(f"   Epochs  : {epochs}  (patience={patience})")
    print(f"   ImgSz   : {imgsz}")
    print(f"   Batch   : {batch}")
    print(f"   Device  : {device}")
    print("=" * 60)

    model = YOLO(model_name)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=project,
        name=name,
        device=device,
        resume=resume,
        workers=workers,
        # Augmentation parameters matching the original Colab run
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        close_mosaic=10,
        erasing=0.4,
        amp=True,       # Automatic mixed precision (speeds up GPU training)
    )

    # Locate the saved best.pt
    best_pt = os.path.abspath(str(results.save_dir) + "/weights/best.pt")

    print("\n" + "=" * 60)
    print(f"✅ Training complete!  Best weights → {best_pt}")
    print("=" * 60)

    # Optionally copy best.pt to the top-level weights/ directory
    os.makedirs(DEFAULT_WEIGHTS_OUT, exist_ok=True)
    dest = os.path.join(DEFAULT_WEIGHTS_OUT, "best.pt")
    shutil.copy2(best_pt, dest)
    print(f"📁 Copied best.pt → {dest}")

    # Optionally export to TFLite
    if export:
        print("\n📦 Exporting to TFLite...")
        best_model = YOLO(best_pt)

        # FP16 export
        best_model.export(format="tflite", imgsz=imgsz)
        print("✅ FP16 TFLite exported.")

        # INT8 export
        best_model.export(format="tflite", int8=True, imgsz=imgsz)
        print("✅ INT8 TFLite exported.")

    return best_pt


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8-seg for document corner detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to the YOLO dataset YAML file.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-seg.pt",
        help="YOLOv8 model variant (e.g. yolov8n-seg.pt, yolov8s-seg.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training and inference image size (pixels).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Mini-batch size. Reduce if VRAM is limited.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help='Training device: "0" (GPU), "cpu", or "0,1" (multi-GPU).',
    )
    parser.add_argument(
        "--project",
        default="runs/segment/DocScan",
        help="Ultralytics output project directory.",
    )
    parser.add_argument(
        "--name",
        default="android_model",
        help="Run name within the project directory.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="DataLoader worker threads (use 0 on Windows if issues arise).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent interrupted training run.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export best.pt to FP16 and INT8 TFLite after training.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for the training pipeline."""
    args = _parse_args()
    run_training(
        data=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        export=args.export,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()

