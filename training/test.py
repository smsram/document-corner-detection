"""
test.py
=======
Evaluation script for the trained YOLOv8-seg document corner detection model.

Runs the Ultralytics validation pipeline on the held-out Mendeley benchmark
and generates:
* Standard YOLO segmentation metrics (Mask mAP50, mAP50-95, Box mAP50, etc.)
* Per-image visualisation grids saved to ``runs/eval/<run_name>/``
* A JSON metrics summary for downstream reporting

Usage
-----
From the repository root::

    # Quick evaluation with default settings
    python training/test.py --data training/dataset.yaml

    # Custom weights and confidence threshold
    python training/test.py \\
        --weights weights/best.pt \\
        --data training/dataset.yaml \\
        --conf 0.25 \\
        --imgsz 640

    # Visualise predictions on specific images
    python training/test.py --data training/dataset.yaml --visualise

    # Save a prediction grid for a directory of images
    python training/test.py \\
        --data training/dataset.yaml \\
        --visualise \\
        --vis-dir assets/ \\
        --vis-count 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# Allow running from either repo root or training/
_TRAIN_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _TRAIN_DIR.parent
if str(_ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR / "src"))

from postprocess import draw_document_overlay, get_strict_4_corners

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = str(_ROOT_DIR / "weights" / "best.pt")
DEFAULT_DATA    = str(_TRAIN_DIR / "dataset.yaml")
DEFAULT_OUT     = str(_ROOT_DIR / "runs" / "eval")


# ===========================================================================
# Metric extraction helper
# ===========================================================================

def _extract_metrics(val_results) -> dict:
    """Extract a flat metrics dictionary from an Ultralytics results object.

    Parameters
    ----------
    val_results : ultralytics.engine.results.Results
        Object returned by ``model.val()``.

    Returns
    -------
    metrics : dict
        Flat dictionary of metric name → float value.
    """
    seg  = val_results.seg
    box  = val_results.box
    return {
        "mask_mAP50":    float(seg.map50),
        "mask_mAP50_95": float(seg.map),
        "mask_precision": float(seg.mp),
        "mask_recall":    float(seg.mr),
        "box_mAP50":    float(box.map50),
        "box_mAP50_95": float(box.map),
        "box_precision": float(box.mp),
        "box_recall":    float(box.mr),
    }


# ===========================================================================
# Visualisation
# ===========================================================================

def _run_visual_eval(
    model: YOLO,
    image_paths: list[str],
    out_dir: Path,
    conf: float,
    imgsz: int,
    n_cols: int = 3,
) -> None:
    """Generate and save a prediction grid for a list of images.

    For each image the function:
    1. Runs YOLOv8-seg inference with ``retina_masks=True``.
    2. Selects the largest detected polygon as the primary document.
    3. Applies ``get_strict_4_corners`` to extract exactly 4 corners.
    4. Draws a semi-transparent quadrangle and corner anchors.

    Parameters
    ----------
    model : YOLO
        Loaded YOLOv8-seg model.
    image_paths : list[str]
        Paths to source images.
    out_dir : Path
        Directory where the prediction grid PNG is saved.
    conf : float
        Detection confidence threshold.
    imgsz : int
        Inference image resolution.
    n_cols : int
        Number of columns in the saved grid figure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(image_paths)
    n_rows = max(1, (n + n_cols - 1) // n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
    axes = np.array(axes).flatten()

    for idx, img_path in enumerate(image_paths):
        preds = model.predict(
            source=img_path,
            imgsz=imgsz,
            conf=conf,
            retina_masks=True,
            verbose=False,
        )[0]

        bgr = cv2.imread(img_path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if preds.masks is not None and len(preds.masks.xy) > 0:
            valid = [m for m in preds.masks.xy if len(m) >= 4]
            if valid:
                largest = max(valid, key=lambda m: cv2.contourArea(m.astype(np.int32)))
                pts = largest.astype(np.int32).reshape((-1, 1, 2))
                corners = get_strict_4_corners(pts)

                # Draw overlay and convert back to RGB for matplotlib
                annotated_bgr = draw_document_overlay(bgr, corners)
                rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(rgb)
        axes[idx].set_title(Path(img_path).name, fontsize=9)
        axes[idx].axis("off")

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()
    grid_path = out_dir / "prediction_grid.png"
    plt.savefig(grid_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"   📊 Prediction grid saved → {grid_path}")


# ===========================================================================
# Main evaluation function
# ===========================================================================

def run_evaluation(
    weights_path: str = DEFAULT_WEIGHTS,
    data_yaml: str = DEFAULT_DATA,
    conf: float = 0.25,
    imgsz: int = 640,
    out_dir: str = DEFAULT_OUT,
    run_name: str = "run",
    visualise: bool = False,
    vis_dir: str | None = None,
    vis_count: int = 8,
    device: str = "0",
) -> dict:
    """Run full model evaluation and optional visualisation.

    Parameters
    ----------
    weights_path : str
        Path to the ``.pt`` checkpoint.
    data_yaml : str
        Path to the YOLO dataset YAML file.
    conf : float
        Detection confidence threshold.
    imgsz : int
        Inference image resolution (pixels).
    out_dir : str
        Base directory for evaluation outputs.
    run_name : str
        Sub-directory name within ``out_dir`` for this run's artefacts.
    visualise : bool
        If True, generate prediction visualisations.
    vis_dir : str | None
        Directory of images to visualise.  Defaults to the YOLO val partition.
    vis_count : int
        Maximum number of images to include in the visualisation grid.
    device : str
        Compute device: ``"0"`` (GPU), ``"cpu"``, ``"mps"``.

    Returns
    -------
    metrics : dict
        Dictionary of metric name → float value.

    Raises
    ------
    FileNotFoundError
        If ``weights_path`` or ``data_yaml`` does not exist.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Place best.pt in the weights/ directory or pass --weights."
        )
    if not os.path.isfile(data_yaml):
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Run data_prep/build_yolo_format.py first, or point --data to your YAML."
        )

    run_out = Path(out_dir) / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    print(f"📦 Loading model: {weights_path}")
    model = YOLO(weights_path)

    # -------------------------------------------------------------------
    # 1. Ultralytics validation metrics
    # -------------------------------------------------------------------
    print(f"\n📊 Running validation on: {data_yaml}")
    val_results = model.val(
        data=data_yaml,
        split="val",
        imgsz=imgsz,
        conf=conf,
        device=device,
        verbose=True,
    )

    metrics = _extract_metrics(val_results)

    # Print formatted table
    print("\n" + "=" * 50)
    print("   Validation Metrics")
    print("=" * 50)
    print(f"   Mask mAP50       : {metrics['mask_mAP50']:.4f}")
    print(f"   Mask mAP50-95    : {metrics['mask_mAP50_95']:.4f}")
    print(f"   Mask Precision   : {metrics['mask_precision']:.4f}")
    print(f"   Mask Recall      : {metrics['mask_recall']:.4f}")
    print("-" * 50)
    print(f"   Box  mAP50       : {metrics['box_mAP50']:.4f}")
    print(f"   Box  mAP50-95    : {metrics['box_mAP50_95']:.4f}")
    print(f"   Box  Precision   : {metrics['box_precision']:.4f}")
    print(f"   Box  Recall      : {metrics['box_recall']:.4f}")
    print("=" * 50)

    # Save metrics to JSON
    metrics_path = run_out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n💾 Metrics saved → {metrics_path}")

    # -------------------------------------------------------------------
    # 2. Optional visualisation
    # -------------------------------------------------------------------
    if visualise:
        # Determine image list
        if vis_dir and os.path.isdir(vis_dir):
            candidates = sorted(
                [str(p) for p in Path(vis_dir).glob("*.png")]
                + [str(p) for p in Path(vis_dir).glob("*.jpg")]
                + [str(p) for p in Path(vis_dir).glob("*.jpeg")]
            )
        else:
            # Fall back to validation images from the YOLO dataset
            import yaml as _yaml  # lightweight; only imported if visualising
            with open(data_yaml) as fh:
                cfg = _yaml.safe_load(fh)
            dataset_root = Path(cfg.get("path", "."))
            val_imgs_dir = dataset_root / cfg.get("val", "images/val")
            candidates = sorted(
                [str(p) for p in val_imgs_dir.glob("*.jpg")]
                + [str(p) for p in val_imgs_dir.glob("*.png")]
            )

        if not candidates:
            print("   ⚠️  No images found for visualisation.")
        else:
            sample = random.sample(candidates, min(vis_count, len(candidates)))
            print(f"\n🖼️  Generating prediction grid for {len(sample)} images…")
            _run_visual_eval(model, sample, run_out / "visuals", conf, imgsz)

    return metrics


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8-seg document corner detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", "-w",
        default=DEFAULT_WEIGHTS,
        help="Path to the .pt checkpoint.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Path to the YOLO dataset YAML.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference resolution (pixels).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT,
        help="Base directory for evaluation outputs.",
    )
    parser.add_argument(
        "--run-name",
        default="run",
        help="Sub-directory name for this evaluation run.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help='Compute device: "0" (GPU), "cpu", "mps".',
    )
    parser.add_argument(
        "--visualise",
        action="store_true",
        help="Generate a prediction grid in addition to metrics.",
    )
    parser.add_argument(
        "--vis-dir",
        default=None,
        help="Directory of images to visualise (defaults to YOLO val set).",
    )
    parser.add_argument(
        "--vis-count",
        type=int,
        default=8,
        help="Number of images to include in the visualisation grid.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for model evaluation."""
    args = _parse_args()
    run_evaluation(
        weights_path=args.weights,
        data_yaml=args.data,
        conf=args.conf,
        imgsz=args.imgsz,
        out_dir=args.out_dir,
        run_name=args.run_name,
        visualise=args.visualise,
        vis_dir=args.vis_dir,
        vis_count=args.vis_count,
        device=args.device,
    )


if __name__ == "__main__":
    main()

