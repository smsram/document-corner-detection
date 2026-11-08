"""
build_yolo_format.py
====================
Assemble the unified YOLO segmentation dataset from all raw downloaded sources.

This script replaces Colab cells 7–10 from ``Paper_Corner.ipynb``.
It reads every raw dataset directory produced by ``download_datasets.py``,
converts all annotation formats to the YOLO-seg polygon label format, and
writes a clean ``yolo_dataset/`` ready for ``training/train.py``.

Additionally synthesises two edge-case categories:
* **Adjacent pages** — a primary document with a partially visible secondary
  page, teaching the model to pick the dominant one.
* **Full-bleed** — a document cropped to its interior content, teaching the
  model that the page can fill the entire camera view.

Usage
-----
From the repository root::

    python data_prep/build_yolo_format.py

    # Override default directories
    python data_prep/build_yolo_format.py \\
        --raw-dir raw_datasets \\
        --out-dir yolo_dataset \\
        --syn-count 1300

Outputs
-------
::

    yolo_dataset/
    ├── images/
    │   ├── train/    ← 5 900+ training images (all sources except Mendeley)
    │   └── val/      ← 1 100+ validation images (Mendeley only)
    ├── labels/
    │   ├── train/    ← YOLO-seg .txt files  (class_id x1 y1 x2 y2 …)
    │   └── val/
    └── dataset.yaml  ← Auto-generated dataset configuration

YOLO Segmentation Label Format
-------------------------------
Each ``.txt`` file contains one line per instance::

    <class_id> <x1> <y1> <x2> <y2> … <xN> <yN>

All coordinates are **normalised** to [0, 1] relative to image width/height.
``class_id`` is always ``0`` (single-class: ``document``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Resolve repository root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
DEFAULT_RAW = str(_ROOT_DIR / "raw_datasets")
DEFAULT_OUT = str(_ROOT_DIR / "yolo_dataset")


# ===========================================================================
# YOLO label writer
# ===========================================================================

def write_yolo_label(
    label_path: str | Path,
    coords: Sequence[Sequence[float]],
    img_w: float = 1.0,
    img_h: float = 1.0,
    is_normalised: bool = False,
) -> None:
    """Write a single-instance YOLO segmentation label file.

    Parameters
    ----------
    label_path : str | Path
        Destination ``.txt`` file path.
    coords : sequence of [x, y]
        Polygon vertices.  Can be absolute pixels or already normalised [0,1].
    img_w, img_h : float
        Image dimensions in pixels.  Used only when ``is_normalised=False``.
    is_normalised : bool
        If True, ``coords`` are already in [0, 1] and no division is applied.

    Notes
    -----
    Coordinates are clamped to [0, 1] after normalisation to guard against
    minor label errors in source datasets.
    """
    flat: list[str] = []
    for pt in coords:
        x, y = float(pt[0]), float(pt[1])
        if not is_normalised:
            x /= img_w
            y /= img_h
        flat.append(f"{max(0.0, min(1.0, x)):.5f}")
        flat.append(f"{max(0.0, min(1.0, y)):.5f}")

    Path(label_path).write_text(f"0 {' '.join(flat)}\n")


# ===========================================================================
# Source converters
# ===========================================================================

def _build_doc_corner(raw_dir: str, train_img: Path, train_lbl: Path) -> int:
    """Convert DocCornerDataset annotations (normalised [0,1]) to YOLO format.

    Parameters
    ----------
    raw_dir : str
        Path to ``raw_datasets/doc_corner/``.
    train_img, train_lbl : Path
        YOLO train image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    ann_path = Path(raw_dir) / "annotations.json"
    if not ann_path.exists():
        print(f"   ⚠️  DocCorner annotations not found at {ann_path} — skipping.")
        return 0

    with open(ann_path) as fh:
        data = json.load(fh)

    count = 0
    for img_name, meta in tqdm(data.items(), desc="1/7 DocCorner"):
        src = Path(raw_dir) / "images" / img_name
        if not src.exists():
            continue
        dst_name = f"doccorner_{img_name}"
        shutil.copy(src, train_img / dst_name)
        write_yolo_label(
            train_lbl / (Path(dst_name).stem + ".txt"),
            meta["corners"],
            is_normalised=True,
        )
        count += 1
    return count


def _build_doc3d(raw_dir: str, train_img: Path, train_lbl: Path) -> int:
    """Extract YOLO polygon labels from Doc3D binary masks via contour analysis.

    For warped/curved pages the mask contour is a faithful description of the
    document boundary.  ``approxPolyDP`` with a tight epsilon (0.5% of arc
    length) keeps enough vertices to represent the curve without excessive
    noise, while still being much smaller than the raw contour.

    Parameters
    ----------
    raw_dir : str
        Path to ``raw_datasets/doc3d/``.
    train_img, train_lbl : Path
        YOLO train image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    mask_files = glob.glob(str(Path(raw_dir) / "masks" / "*.png"))
    count = 0

    for mask_path in tqdm(mask_files, desc="2/7 Doc3D"):
        base_id = Path(mask_path).stem
        # Images may be .jpg, .png, or .jpeg
        matches = glob.glob(str(Path(raw_dir) / "images" / f"{base_id}.*"))
        if not matches:
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape[:2]
        # Find the outer boundary of the paper region
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)

        # RDP approximation: 0.5% of arc length keeps curve fidelity
        # without thousands of vertices
        epsilon = 0.005 * cv2.arcLength(largest, closed=True)
        approx = cv2.approxPolyDP(largest, epsilon, closed=True)
        coords = [pt[0].tolist() for pt in approx]

        if len(coords) < 3:
            continue

        dst_name = f"{base_id}.jpg"
        shutil.copy(matches[0], train_img / dst_name)
        write_yolo_label(
            train_lbl / (Path(dst_name).stem + ".txt"),
            coords, w, h,
        )
        count += 1
    return count


def _build_midv500(raw_dir: str, train_img: Path, train_lbl: Path) -> int:
    """Convert MIDV-500 absolute quad coordinates to YOLO format.

    Ground-truth quads are stored as JSON files alongside each image in
    ``ground_truth/`` with the same base name.

    Parameters
    ----------
    raw_dir : str
        Path to ``raw_datasets/midv500/``.
    train_img, train_lbl : Path
        YOLO train image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    img_files = glob.glob(str(Path(raw_dir) / "**" / "*.tif"), recursive=True)
    count = 0

    for src_path in tqdm(img_files, desc="3/7 MIDV-500"):
        json_path = src_path.replace("/images/", "/ground_truth/").replace(".tif", ".json")
        if not os.path.exists(json_path):
            continue

        with open(json_path) as fh:
            jdata = json.load(fh)
        quad = jdata.get("quad")
        if not quad:
            continue

        img = cv2.imread(src_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        dst_name = f"midv_{Path(src_path).stem}.jpg"
        cv2.imwrite(str(train_img / dst_name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        write_yolo_label(train_lbl / (Path(dst_name).stem + ".txt"), quad, w, h)
        count += 1
    return count


def _build_smartdoc15(raw_dir: str, train_img: Path, train_lbl: Path) -> int:
    """Convert SmartDoc 2015 absolute quad coordinates to YOLO format.

    Parameters
    ----------
    raw_dir : str
        Path to ``raw_datasets/smartdoc15/``.
    train_img, train_lbl : Path
        YOLO train image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    ann_path = Path(raw_dir) / "annotations.json"
    if not ann_path.exists():
        print(f"   ⚠️  SmartDoc annotations not found — skipping.")
        return 0

    with open(ann_path) as fh:
        data = json.load(fh)

    count = 0
    for img_name, meta in tqdm(data.items(), desc="4/7 SmartDoc"):
        src = Path(raw_dir) / "images" / img_name
        if not src.exists():
            continue
        dst_name = f"smartdoc_{img_name}"
        shutil.copy(src, train_img / dst_name)
        write_yolo_label(
            train_lbl / (Path(dst_name).stem + ".txt"),
            meta["corners"], meta["width"], meta["height"],
        )
        count += 1
    return count


def _build_cord(raw_dir: str, train_img: Path, train_lbl: Path) -> int:
    """Write full-bleed (image-boundary) labels for all CORD-v2 images.

    CORD-v2 receipts fill the entire camera frame, so the document quad
    equals the image boundary: (0,0) → (W,0) → (W,H) → (0,H).

    Parameters
    ----------
    raw_dir : str
        Path to ``raw_datasets/cord/``.
    train_img, train_lbl : Path
        YOLO train image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    img_files = glob.glob(str(Path(raw_dir) / "images" / "*.jpg"))
    count = 0

    for src_path in tqdm(img_files, desc="5/7 CORD-v2"):
        img = cv2.imread(src_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        dst_name = f"cord_{Path(src_path).name}"
        shutil.copy(src_path, train_img / dst_name)
        # Full-bleed: quad = image corners
        write_yolo_label(
            train_lbl / (Path(dst_name).stem + ".txt"),
            [[0, 0], [w, 0], [w, h], [0, h]], w, h,
        )
        count += 1
    return count


def _build_synthetic(
    doclaynet_dir: str,
    cord_dir: str,
    syn_out: Path,
    n_adjacent: int = 800,
    n_fullbleed: int = 500,
) -> tuple[Path, Path]:
    """Generate synthetic edge-case images and save them to ``syn_out``.

    Two categories are generated:

    **Adjacent pages** (``n_adjacent`` images)
        A primary document is placed on a coloured background canvas; a
        secondary page (the distractor) is pasted first so it appears
        *behind* the primary.  The ground-truth quad is the primary page only.

    **Full-bleed** (``n_fullbleed`` images)
        A random centre crop of a source document is resized to 800×800,
        training the model to treat the image boundary as the document corners.

    Parameters
    ----------
    doclaynet_dir : str
        Path to ``raw_datasets/doclaynet/images/``.
    cord_dir : str
        Path to ``raw_datasets/cord/images/``.
    syn_out : Path
        Output directory for generated images and annotations.
    n_adjacent : int
        Number of adjacent-page composites to generate.
    n_fullbleed : int
        Number of full-bleed crops to generate.

    Returns
    -------
    img_dir : Path
        Path to generated images sub-directory.
    ann_path : Path
        Path to the generated ``annotations.json``.
    """
    img_dir = syn_out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Collect source document textures from DocLayNet and CORD
    source_docs = (
        glob.glob(str(Path(doclaynet_dir) / "*.jpg")) +
        glob.glob(str(Path(cord_dir) / "*.jpg"))
    )

    if len(source_docs) < 10:
        raise ValueError(
            "Not enough source images for synthetic generation. "
            "Ensure DocLayNet and CORD are downloaded."
        )

    annotations: dict = {}
    random.seed(42)

    # ------------------------------------------------------------------
    # Part A: Adjacent / Overlapping Pages
    # ------------------------------------------------------------------
    print("   🎨 Generating adjacent-page composites…")
    for i in tqdm(range(n_adjacent), desc="Synthetic:Adjacent"):
        canvas_w, canvas_h = 1024, 1024

        # Solid-colour desk background
        bg = (random.randint(40, 200), random.randint(40, 200), random.randint(40, 200))
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg)

        # Primary document (target)
        try:
            doc_img = Image.open(random.choice(source_docs)).convert("RGB")
        except Exception:
            continue

        doc_w = random.randint(450, 600)
        doc_h = int(doc_w * 1.414)  # A4 aspect ratio
        doc_img = doc_img.resize((doc_w, doc_h))

        main_x = random.randint(150, max(151, canvas_w - doc_w - 50))
        main_y = random.randint(100, max(101, canvas_h - doc_h - 50))

        # Distractor page (placed first → appears behind primary)
        try:
            adj_img = Image.open(random.choice(source_docs)).convert("RGB").resize((doc_w, doc_h))
        except Exception:
            continue

        # Distractor touches the left or right edge of the primary
        if random.random() > 0.5:
            adj_x = main_x + doc_w - random.randint(10, 60)   # right overlap
        else:
            adj_x = main_x - doc_w + random.randint(10, 60)   # left overlap
        adj_y = main_y + random.randint(-40, 40)

        canvas.paste(adj_img, (adj_x, adj_y))   # distractor behind
        canvas.paste(doc_img, (main_x, main_y))  # primary on top

        img_name = f"syn_adj_{i:04d}.jpg"
        canvas.save(img_dir / img_name, quality=90)

        # Ground-truth quad: primary document only (normalised)
        annotations[img_name] = {
            "corners": [
                [main_x / canvas_w, main_y / canvas_h],
                [(main_x + doc_w) / canvas_w, main_y / canvas_h],
                [(main_x + doc_w) / canvas_w, (main_y + doc_h) / canvas_h],
                [main_x / canvas_w, (main_y + doc_h) / canvas_h],
            ],
            "type": "adjacent_overlap",
        }

    # ------------------------------------------------------------------
    # Part B: Full-Bleed Crops
    # ------------------------------------------------------------------
    print("   📐 Generating full-bleed crops…")
    for i in tqdm(range(n_fullbleed), desc="Synthetic:FullBleed"):
        try:
            doc_img = Image.open(random.choice(source_docs)).convert("RGB")
        except Exception:
            continue

        w, h = doc_img.size
        crop_w = int(w * random.uniform(0.5, 0.75))
        crop_h = int(h * random.uniform(0.5, 0.75))
        if crop_w < 50 or crop_h < 50:
            continue

        left = random.randint(20, max(21, w - crop_w - 20))
        top = random.randint(20, max(21, h - crop_h - 20))
        cropped = doc_img.crop((left, top, left + crop_w, top + crop_h)).resize((800, 800))

        img_name = f"syn_full_{i:04d}.jpg"
        cropped.save(img_dir / img_name, quality=90)

        # Ground-truth quad: image boundary (normalised [0,1])
        annotations[img_name] = {
            "corners": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "type": "full_bleed",
        }

    ann_path = syn_out / "annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2))
    print(f"   ✅ Synthetic: {len(annotations)} samples generated.")
    return img_dir, ann_path


def _build_synthetic_to_yolo(
    syn_ann_path: Path,
    syn_img_dir: Path,
    train_img: Path,
    train_lbl: Path,
) -> int:
    """Copy synthetic images and write their YOLO labels.

    Parameters
    ----------
    syn_ann_path : Path
        Path to the synthetic ``annotations.json``.
    syn_img_dir : Path
        Path to the synthetic images directory.
    train_img, train_lbl : Path
        YOLO train image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    with open(syn_ann_path) as fh:
        data = json.load(fh)

    count = 0
    for img_name, meta in tqdm(data.items(), desc="6/7 Synthetic"):
        src = syn_img_dir / img_name
        if not src.exists():
            continue
        dst_name = f"synthetic_{img_name}"
        shutil.copy(src, train_img / dst_name)
        write_yolo_label(
            train_lbl / (Path(dst_name).stem + ".txt"),
            meta["corners"],
            is_normalised=True,
        )
        count += 1
    return count


def _build_mendeley_val(raw_dir: str, val_img: Path, val_lbl: Path) -> int:
    """Parse Mendeley CSV quad annotations and write the YOLO validation set.

    Each CSV file contains 4 (x, y) coordinate pairs in absolute pixels.
    The CSV may be shaped as (1 × 8) or (4 × 2) — both are handled by
    flattening to a 1-D array and reshaping to (4, 2).

    Parameters
    ----------
    raw_dir : str
        Path to ``raw_datasets/mendeley/``.
    val_img, val_lbl : Path
        YOLO validation image and label directories.

    Returns
    -------
    count : int
        Number of samples written.
    """
    csv_files = sorted(glob.glob(str(Path(raw_dir) / "**" / "*.csv"), recursive=True))
    count = 0

    for csv_path in tqdm(csv_files, desc="7/7 Mendeley (Val)"):
        img_path = csv_path.replace(".csv", ".jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            df = pd.read_csv(csv_path, header=None)
            vals = df.values.flatten()
            if len(vals) < 8:
                continue

            # Reshape the first 8 flat values into 4 (x, y) pairs
            raw_coords = vals[:8].reshape(4, 2).astype(float)
        except Exception:
            continue

        h, w = img.shape[:2]
        out_stem = f"mendeley_{count:05d}"
        cv2.imwrite(str(val_img / f"{out_stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        write_yolo_label(val_lbl / f"{out_stem}.txt", raw_coords, w, h)
        count += 1
    return count


# ===========================================================================
# Dataset YAML writer
# ===========================================================================

def write_dataset_yaml(out_dir: Path) -> None:
    """Write the ``dataset.yaml`` configuration file for Ultralytics training.

    Parameters
    ----------
    out_dir : Path
        Root of the assembled YOLO dataset (the directory that contains
        ``images/`` and ``labels/``).
    """
    yaml_path = out_dir / "dataset.yaml"
    content = (
        f"# Auto-generated by data_prep/build_yolo_format.py\n"
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"\n"
        f"names:\n"
        f"  0: document\n"
    )
    yaml_path.write_text(content)
    print(f"\n📄 Dataset YAML written → {yaml_path}")


# ===========================================================================
# Main assembly orchestrator
# ===========================================================================

def build_dataset(raw_dir: str, out_dir: str, syn_count: int = 1300) -> None:
    """Assemble the unified YOLO segmentation dataset from all raw sources.

    Parameters
    ----------
    raw_dir : str
        Root directory of raw downloaded datasets (from ``download_datasets.py``).
    out_dir : str
        Destination root for the assembled YOLO dataset.
    syn_count : int
        Total number of synthetic images to generate (split 800 adjacent / 500 full-bleed).
    """
    raw = Path(raw_dir)
    out = Path(out_dir)

    # Create YOLO directory tree
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_img = out / "images" / "train"
    train_lbl = out / "labels" / "train"
    val_img   = out / "images" / "val"
    val_lbl   = out / "labels" / "val"

    totals: dict[str, int] = {}

    print("\n" + "=" * 60)
    print("🔨 Assembling YOLO-seg dataset…")
    print("=" * 60)

    # 1. DocCorner
    n = _build_doc_corner(str(raw / "doc_corner"), train_img, train_lbl)
    totals["DocCorner"] = n

    # 2. Doc3D
    n = _build_doc3d(str(raw / "doc3d"), train_img, train_lbl)
    totals["Doc3D"] = n

    # 3. MIDV-500
    n = _build_midv500(str(raw / "midv500"), train_img, train_lbl)
    totals["MIDV-500"] = n

    # 4. SmartDoc 2015
    n = _build_smartdoc15(str(raw / "smartdoc15"), train_img, train_lbl)
    totals["SmartDoc"] = n

    # 5. CORD-v2 (full-bleed)
    n = _build_cord(str(raw / "cord"), train_img, train_lbl)
    totals["CORD-v2"] = n

    # 6. Synthetic edge cases
    n_adj = int(syn_count * (800 / 1300))
    n_full = syn_count - n_adj
    syn_out = out / "_synthetic"
    try:
        syn_img_dir, syn_ann_path = _build_synthetic(
            doclaynet_dir=str(raw / "doclaynet" / "images"),
            cord_dir=str(raw / "cord" / "images"),
            syn_out=syn_out,
            n_adjacent=n_adj,
            n_fullbleed=n_full,
        )
        n = _build_synthetic_to_yolo(syn_ann_path, syn_img_dir, train_img, train_lbl)
        totals["Synthetic"] = n
    except ValueError as e:
        print(f"   ⚠️  Skipping synthetic generation: {e}")
        totals["Synthetic"] = 0

    # 7. Mendeley validation partition
    n = _build_mendeley_val(str(raw / "mendeley"), val_img, val_lbl)
    totals["Mendeley (Val)"] = n

    # Write dataset.yaml
    write_dataset_yaml(out)

    # Also update training/dataset.yaml to point to this location
    training_yaml = _ROOT_DIR / "training" / "dataset.yaml"
    if training_yaml.exists():
        content = training_yaml.read_text()
        import re
        updated = re.sub(
            r"^path:.*$",
            f"path: {out.resolve()}",
            content,
            flags=re.MULTILINE,
        )
        training_yaml.write_text(updated)
        print(f"📄 Updated training/dataset.yaml path → {out.resolve()}")

    # Summary
    total_train = sum(totals[k] for k in totals if k != "Mendeley (Val)")
    total_val = totals.get("Mendeley (Val)", 0)

    print("\n" + "=" * 60)
    print("✅ UNIFIED YOLO DATASET ASSEMBLY COMPLETE")
    print("=" * 60)
    for src, cnt in totals.items():
        split = "Val " if "Val" in src else "Train"
        print(f"   [{split}]  {src:<18}  {cnt:>5} images")
    print("-" * 60)
    print(f"   Total Training: {total_train}")
    print(f"   Total Validation: {total_val}")
    print(f"\n   Dataset root: {out.resolve()}")
    print("=" * 60)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw downloaded datasets into a unified YOLO segmentation dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        default=DEFAULT_RAW,
        help="Root directory of raw datasets (output of download_datasets.py).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT,
        help="Destination root for the assembled YOLO dataset.",
    )
    parser.add_argument(
        "--syn-count",
        type=int,
        default=1300,
        help="Total number of synthetic images to generate (adjacent + full-bleed).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point."""
    args = _parse_args()
    build_dataset(args.raw_dir, args.out_dir, args.syn_count)


if __name__ == "__main__":
    main()

