"""
export.py
=========
Export a trained YOLOv8-seg PyTorch checkpoint to TensorFlow Lite formats
suitable for Android edge deployment.

Two quantisation variants are produced:
* **FP16** (``best.tflite``)      — balanced speed/accuracy; recommended as a
  starting point.
* **INT8** (``best_int8.tflite``) — maximum throughput on low/mid-range
  Snapdragon or MediaTek chipsets; minor accuracy trade-off.

Usage
-----
From the repository root::

    python src/export.py

    # Custom weights path and image size
    python src/export.py --weights weights/best.pt --imgsz 640

    # Export only INT8 (skip FP16)
    python src/export.py --no-fp16

    # Export only FP16 (skip INT8)
    python src/export.py --no-int8

    # Place outputs in a custom directory
    python src/export.py --out-dir my_exports/

Notes
-----
* Ultralytics places the exported ``.tflite`` files alongside the source
  ``.pt`` checkpoint in a subfolder named after the format.  This script
  copies them to ``--out-dir`` (default: ``weights/``) for convenience.
* TFLite export requires ``tensorflow`` ≥ 2.12 to be installed.  Install it
  separately::

      pip install tensorflow

* INT8 export performs post-training quantisation using a small calibration
  dataset derived from the training images; ensure the dataset path in
  ``dataset.yaml`` is accessible when running ``--no-int8=False``.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

# Allow running directly from either the repo root or src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)

# ---------------------------------------------------------------------------
# Default paths (relative to the repository root)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = os.path.join(_ROOT_DIR, "weights", "best.pt")
DEFAULT_OUT_DIR = os.path.join(_ROOT_DIR, "weights")


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------

def export_tflite(
    weights_path: str = DEFAULT_WEIGHTS,
    imgsz: int = 640,
    out_dir: str = DEFAULT_OUT_DIR,
    export_fp16: bool = True,
    export_int8: bool = True,
) -> list[str]:
    """Export a YOLOv8-seg model to TFLite (FP16 and/or INT8).

    Parameters
    ----------
    weights_path : str
        Path to the source ``.pt`` YOLOv8-seg checkpoint.
    imgsz : int
        Square inference image resolution used during export and at runtime.
        Must match the resolution used during training (default 640).
    out_dir : str
        Directory to copy the exported ``.tflite`` files into.
    export_fp16 : bool
        Whether to export the FP16 (half-precision float) variant.
    export_int8 : bool
        Whether to export the INT8 (quantised integer) variant.

    Returns
    -------
    exported_paths : list[str]
        Absolute paths to all successfully exported ``.tflite`` files.

    Raises
    ------
    FileNotFoundError
        If ``weights_path`` does not exist.
    RuntimeError
        If no export variant was requested.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            "Ensure best.pt is in the weights/ directory or pass --weights."
        )

    if not export_fp16 and not export_int8:
        raise RuntimeError("At least one of --fp16 or --int8 must be enabled.")

    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.abspath(weights_path)
    exported_paths = []

    # ------------------------------------------------------------------
    # FP16 export
    # ------------------------------------------------------------------
    if export_fp16:
        print("=" * 55)
        print("📦 Exporting FP16 TFLite model...")
        print("=" * 55)

        model = YOLO(weights_path)
        # Ultralytics export() returns the path to the saved file/directory
        fp16_result = model.export(
            format="tflite",
            imgsz=imgsz,
        )

        # Locate the generated .tflite file (may be nested in a _saved_model dir)
        fp16_src = _find_tflite(fp16_result, int8=False)
        if fp16_src:
            dst = os.path.join(out_dir, "best.tflite")
            shutil.copy2(fp16_src, dst)
            exported_paths.append(dst)
            print(f"✅ FP16 model saved → {dst}")
        else:
            print(f"⚠️  FP16 export produced no .tflite file at: {fp16_result}")

    # ------------------------------------------------------------------
    # INT8 export
    # ------------------------------------------------------------------
    if export_int8:
        print("=" * 55)
        print("📦 Exporting INT8 quantised TFLite model...")
        print("   (Calibration may take a few minutes)")
        print("=" * 55)

        model = YOLO(weights_path)
        int8_result = model.export(
            format="tflite",
            imgsz=imgsz,
            int8=True,
        )

        int8_src = _find_tflite(int8_result, int8=True)
        if int8_src:
            dst = os.path.join(out_dir, "best_int8.tflite")
            shutil.copy2(int8_src, dst)
            exported_paths.append(dst)
            print(f"✅ INT8 model saved  → {dst}")
        else:
            print(f"⚠️  INT8 export produced no .tflite file at: {int8_result}")

    return exported_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_tflite(export_result, int8: bool) -> str | None:
    """Locate the .tflite output file from an Ultralytics export() result.

    Ultralytics may return a string path, a Path object, or a directory.
    This helper searches for the first matching ``.tflite`` file.

    Parameters
    ----------
    export_result : str | Path | None
        Return value from ``YOLO.export()``.
    int8 : bool
        If True, prefer files containing ``_int8`` in the name.

    Returns
    -------
    path : str | None
        Absolute path to the located ``.tflite`` file, or ``None``.
    """
    if export_result is None:
        return None

    candidate = Path(str(export_result))

    # Direct file reference
    if candidate.is_file() and candidate.suffix == ".tflite":
        return str(candidate)

    # Directory returned — search recursively
    search_root = candidate if candidate.is_dir() else candidate.parent
    tflite_files = sorted(search_root.rglob("*.tflite"))

    if not tflite_files:
        return None

    # Prefer the _int8 variant when requested, otherwise the plain one
    if int8:
        for f in tflite_files:
            if "int8" in f.name:
                return str(f)
    else:
        for f in tflite_files:
            if "int8" not in f.name:
                return str(f)

    # Last resort: return the first found
    return str(tflite_files[0])


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8-seg .pt model to TFLite for Android deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", "-w",
        default=DEFAULT_WEIGHTS,
        help="Path to the YOLOv8-seg .pt checkpoint.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Export and runtime inference image size (pixels).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Directory to place exported .tflite files.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip the FP16 TFLite export.",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Skip the INT8 TFLite export.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for TFLite export."""
    args = _parse_args()

    paths = export_tflite(
        weights_path=args.weights,
        imgsz=args.imgsz,
        out_dir=args.out_dir,
        export_fp16=not args.no_fp16,
        export_int8=not args.no_int8,
    )

    print("\n" + "=" * 55)
    print(f"🎉 Export complete! {len(paths)} file(s) written:")
    for p in paths:
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"   {p}  ({size_mb:.2f} MB)")
    print("=" * 55)


if __name__ == "__main__":
    main()

