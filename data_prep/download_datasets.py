"""
download_datasets.py
====================
Stream and save all training and validation datasets required for the
YOLOv8-seg document corner detection model.

This script replaces Colab cells 1–6 from ``Paper_Corner.ipynb``.
It runs fully locally using standard Python I/O, and authenticates to
Hugging Face via an environment variable so no credentials are hardcoded.

Datasets downloaded
-------------------
1. **DocCornerDataset** (mapo80)         — 2 000 baseline flat-document images
2. **CORD-v2**  (naver-clova-ix)         — 800  full-bleed receipt images
3. **DocLayNet-v1.2** (docling-project)  — 1 500 digital-layout images (texture only)
4. **Doc3D** (StonyBrook-CVLab)          — 1 500 warped/curved paper images + albedo masks
5. **MIDV-500** (SmartEngines FTP)       — 400  identity-document images with quads
6. **SmartDoc 2015** (GitHub release)    — ~30  extreme-perspective video frames
7. **Mendeley Corner** (Mendeley API)    — ~1 100 smartphone captures (validation only)

Usage
-----
From the repository root::

    # Set your Hugging Face token first (required for gated repos)
    export HF_TOKEN="hf_..."         # Linux / macOS
    set  HF_TOKEN=hf_...             # Windows CMD
    $env:HF_TOKEN="hf_..."           # Windows PowerShell

    python data_prep/download_datasets.py

    # Download only specific datasets
    python data_prep/download_datasets.py --datasets doc_corner cord mendeley

    # Override output directory
    python data_prep/download_datasets.py --out-dir /data/raw_datasets

Outputs
-------
All raw data is written to ``raw_datasets/`` (relative to the repo root)::

    raw_datasets/
    ├── doc_corner/
    │   ├── images/           ← JPEG images
    │   └── annotations.json  ← {filename: {width, height, corners: [[x,y]×4]}}
    ├── cord/
    │   ├── images/
    │   └── annotations.json
    ├── doclaynet/
    │   └── images/           ← No annotation needed (texture source only)
    ├── doc3d/
    │   ├── images/
    │   └── masks/            ← Binary PNG masks (white=paper, black=background)
    ├── midv500/              ← MIDV zip extracted in-place (images + ground_truth/)
    ├── smartdoc15/
    │   ├── images/
    │   └── annotations.json
    └── mendeley/             ← Mendeley zip extracted in-place (images + .csv quads)
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Resolve repository root (works whether cwd is repo root or data_prep/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SCRIPT_DIR.parent
DEFAULT_OUT = str(_ROOT_DIR / "raw_datasets")

# ---------------------------------------------------------------------------
# All available dataset keys (used by --datasets flag)
# ---------------------------------------------------------------------------
ALL_DATASETS = [
    "doc_corner",
    "cord",
    "doclaynet",
    "doc3d",
    "midv500",
    "smartdoc15",
    "mendeley",
]


# ===========================================================================
# Hugging Face helpers
# ===========================================================================

def _get_hf_token() -> str | None:
    """Read the Hugging Face access token from the environment.

    Returns ``None`` when no token is set (public datasets only).
    Set ``HF_TOKEN`` in your shell before running::

        export HF_TOKEN="hf_..."
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "⚠️  HF_TOKEN not set. Public datasets will work, but gated repos\n"
            "   (Doc3D, DocLayNet) require authentication.\n"
            "   Set the variable and re-run:\n"
            "       export HF_TOKEN='hf_...'   # Linux/macOS\n"
            "       $env:HF_TOKEN='hf_...'     # PowerShell\n"
        )
    return token


def _hf_login(token: str | None) -> None:
    """Authenticate the ``huggingface_hub`` client with ``token``."""
    from huggingface_hub import login
    if token:
        login(token=token, add_to_git_credential=False)
        print("✅ Hugging Face authenticated.")


# ===========================================================================
# 1. DocCornerDataset (mapo80)
# ===========================================================================

def download_doc_corner(out_dir: str, max_images: int = 2000) -> None:
    """Stream and save up to ``max_images`` images from mapo80/DocCornerDataset.

    Each item contains normalised corner coordinates [TL, TR, BR, BL]
    stored directly as HuggingFace dataset fields.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/doc_corner``).
    max_images : int
        Maximum number of non-negative samples to retain.
    """
    from datasets import load_dataset

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 1/7  Streaming DocCornerDataset (max {max_images})…")
    ds = load_dataset("mapo80/DocCornerDataset", split="train", streaming=True)

    annotations: dict = {}
    count = 0

    for item in tqdm(ds, total=max_images, desc="DocCorner"):
        # Skip negative samples (no document present)
        if item.get("is_negative", False):
            continue

        img_id = f"corner_{count:05d}"
        img_path = img_dir / f"{img_id}.jpg"

        pil_img: Image.Image = item["image"].convert("RGB")
        w, h = pil_img.size
        pil_img.save(img_path, quality=90)

        # Normalised [0, 1] corner coordinates stored per-field in the dataset
        annotations[f"{img_id}.jpg"] = {
            "width": w,
            "height": h,
            "corners": [
                [item["corner_tl_x"], item["corner_tl_y"]],  # Top-Left
                [item["corner_tr_x"], item["corner_tr_y"]],  # Top-Right
                [item["corner_br_x"], item["corner_br_y"]],  # Bottom-Right
                [item["corner_bl_x"], item["corner_bl_y"]],  # Bottom-Left
            ],
        }

        count += 1
        if count >= max_images:
            break

    ann_path = Path(out_dir) / "annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2))
    print(f"   ✅ DocCorner: saved {count} images → {out_dir}")


# ===========================================================================
# 2. CORD-v2 (naver-clova-ix)
# ===========================================================================

def download_cord(out_dir: str, max_images: int = 800) -> None:
    """Stream and save up to ``max_images`` receipt images from CORD-v2.

    CORD-v2 images fill the full camera view, so their bounding quad is
    always the image boundary — no corner annotation is extracted here.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/cord``).
    max_images : int
        Maximum number of images to save.
    """
    from datasets import load_dataset

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 2/7  Streaming CORD-v2 (max {max_images})…")
    ds = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)

    annotations: dict = {}
    count = 0

    for item in tqdm(ds, total=max_images, desc="CORD-v2"):
        img_id = f"cord_{count:05d}"
        img_path = img_dir / f"{img_id}.jpg"

        pil_img: Image.Image = item["image"].convert("RGB")
        w, h = pil_img.size
        pil_img.save(img_path, quality=90)

        # Ground truth is stored as a JSON string — parse it
        gt = item.get("ground_truth")
        if isinstance(gt, str):
            gt = json.loads(gt)

        # Save the raw GT so build_yolo_format.py can assign full-bleed corners
        annotations[f"{img_id}.jpg"] = {"width": w, "height": h, "raw_gt": gt}

        count += 1
        if count >= max_images:
            break

    ann_path = Path(out_dir) / "annotations.json"
    ann_path.write_text(json.dumps(annotations, indent=2))
    print(f"   ✅ CORD-v2:  saved {count} images → {out_dir}")


# ===========================================================================
# 3. DocLayNet-v1.2 (texture source for synthetic data)
# ===========================================================================

def download_doclaynet(out_dir: str, max_images: int = 1500) -> None:
    """Stream and save up to ``max_images`` images from DocLayNet-v1.2.

    DocLayNet images are used **only as texture backgrounds** for the
    synthetic adjacent-page generator in ``build_yolo_format.py``.
    No annotation is saved.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/doclaynet``).
    max_images : int
        Maximum number of images to save.
    """
    from datasets import load_dataset

    img_dir = Path(out_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📥 3/7  Streaming DocLayNet-v1.2 (max {max_images})…")
    ds = load_dataset(
        "docling-project/DocLayNet-v1.2",
        name="default",
        split="train",
        streaming=True,
    )

    count = 0
    for item in tqdm(ds, total=max_images, desc="DocLayNet"):
        pil_img: Image.Image = item["image"].convert("RGB")
        pil_img.save(img_dir / f"doclaynet_{count:05d}.jpg", quality=85)
        count += 1
        if count >= max_images:
            break

    print(f"   ✅ DocLayNet: saved {count} images → {out_dir}")


# ===========================================================================
# 4. Doc3D — warped paper with albedo masks
# ===========================================================================

def download_doc3d(out_dir: str, max_images: int = 1500, hf_token: str | None = None) -> None:
    """Download and extract image/albedo pairs from StonyBrook-CVLab/doc3D-dataset.

    The albedo (colour-corrected render) pass has a strict black background
    ([0, 0, 0]) everywhere there is no paper.  A threshold of 15 on the max
    channel reliably separates paper from void, producing pixel-exact binary
    masks even for curved pages.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/doc3d``).
    max_images : int
        Maximum number of image–mask pairs to extract.
    hf_token : str | None
        Hugging Face token (required — this dataset is gated).
    """
    from huggingface_hub import HfApi, hf_hub_download

    img_dir = Path(out_dir) / "images"
    mask_dir = Path(out_dir) / "masks"
    tmp_dir = Path(out_dir) / "_tmp"

    # Wipe stale directories for a clean run
    for d in (img_dir, mask_dir, tmp_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    repo_id = "StonyBrook-CVLab/doc3D-dataset"
    print(f"\n📥 4/7  Downloading Doc3D from {repo_id} (max {max_images})…")

    api = HfApi(token=hf_token)
    all_files = list(api.list_repo_files(repo_id, repo_type="dataset"))

    # Locate the first image zip and its matching albedo zip (img_1.zip → alb_1.zip)
    img_zips = [f for f in all_files if re.search(r"img_\d+\.zip", f, re.I)]
    if not img_zips:
        raise ValueError("No image zips found in doc3D-dataset repository.")

    target_img_zip = img_zips[0]
    target_alb_zip = target_img_zip.replace("img_", "alb_")
    if target_alb_zip not in all_files:
        alb_zips = [f for f in all_files if re.search(r"alb_\d+\.zip", f, re.I)]
        target_alb_zip = alb_zips[0] if alb_zips else None

    if not target_alb_zip:
        raise ValueError("No albedo (alb_*.zip) found in doc3D-dataset repository.")

    print(f"   Downloading image zip: {target_img_zip}")
    img_zip_path = hf_hub_download(
        repo_id=repo_id, filename=target_img_zip,
        repo_type="dataset", token=hf_token, local_dir=str(tmp_dir),
    )

    print(f"   Downloading albedo zip: {target_alb_zip}")
    alb_zip_path = hf_hub_download(
        repo_id=repo_id, filename=target_alb_zip,
        repo_type="dataset", token=hf_token, local_dir=str(tmp_dir),
    )

    success = 0
    print(f"   Extracting up to {max_images} image/mask pairs…")

    with zipfile.ZipFile(img_zip_path) as z_img, zipfile.ZipFile(alb_zip_path) as z_alb:
        # Build a lookup from relative key → albedo member name for fast matching
        alb_map: dict[str, str] = {}
        for m in z_alb.namelist():
            if m.lower().endswith((".png", ".jpg", ".jpeg")):
                rel_key = m.replace("alb/", "").replace("img/", "").strip("/")
                alb_map[rel_key] = m
                base_key = Path(m).stem
                alb_map[base_key] = m

        img_members = [
            f for f in z_img.namelist()
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for img_member in tqdm(img_members[:max_images], desc="Doc3D pairs"):
            img_id = f"doc3d_{success:05d}"

            # --- Save RGB image ---
            with z_img.open(img_member) as fh:
                Image.open(fh).convert("RGB").save(
                    img_dir / f"{img_id}.jpg", quality=90
                )

            # --- Match and threshold albedo pass → binary mask ---
            rel_key = img_member.replace("img/", "").replace("alb/", "").strip("/")
            base_key = Path(img_member).stem
            alb_member = alb_map.get(rel_key) or alb_map.get(base_key)

            if alb_member:
                with z_alb.open(alb_member) as fh:
                    buf = np.asarray(bytearray(fh.read()), dtype=np.uint8)
                    # Decode strictly as BGR; avoids alpha-channel corruption
                    alb_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)

                if alb_bgr is not None:
                    # Albedo background is pitch-black [0,0,0]; paper has colour >15
                    mask = (np.max(alb_bgr, axis=2) > 15).astype(np.uint8) * 255
                    # Morphological closing seals minor edge-aliasing gaps
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    cv2.imwrite(str(mask_dir / f"{img_id}.png"), mask)
            else:
                # Fallback: luminance threshold on the saved image
                img_bgr = cv2.imread(str(img_dir / f"{img_id}.jpg"))
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
                cv2.imwrite(str(mask_dir / f"{img_id}.png"), mask)

            success += 1

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"   ✅ Doc3D: {success} pairs → {out_dir}")


# ===========================================================================
# 5. MIDV-500
# ===========================================================================

def download_midv500(out_dir: str, max_images: int = 400) -> None:
    """Download and prune the MIDV-500 identity document dataset.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/midv500``).
    max_images : int
        Maximum number of image–annotation pairs to retain.
    """
    import glob as _glob

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    zip_path = base / "midv_01.zip"
    ftp_url = "ftp://smartengines.com/midv-500/dataset/01_alb_id.zip"

    print(f"\n📥 5/7  Downloading MIDV-500 from SmartEngines FTP…")

    # Attempt FTP download via curl (most reliable cross-platform option)
    exit_code = os.system(
        f'curl -f -# -o "{zip_path}" "{ftp_url}"'
    )

    if exit_code != 0 or not zip_path.exists() or zip_path.stat().st_size < 10_000:
        print("   ⚠️  FTP blocked. Trying midv500 Python package as fallback…")
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "midv500"]
            )
            import midv500  # type: ignore
            midv500.download_dataset(str(base), "01_alb_id")
        except Exception as e:
            print(f"   ❌ MIDV-500 download failed: {e}")
            print("   Please download manually from https://smartengines.com/midv-500/")
            return
    else:
        print("   Extracting archive…")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(base)
        zip_path.unlink()

    # Prune to max_images keeping image+json pairs synchronised
    print(f"   Pruning to {max_images} samples…")
    img_files = sorted(_glob.glob(str(base / "**" / "*.tif"), recursive=True))

    for i, img_path in enumerate(img_files):
        if i >= max_images:
            os.remove(img_path)
            json_path = img_path.replace("/images/", "/ground_truth/").replace(".tif", ".json")
            if os.path.exists(json_path):
                os.remove(json_path)

    remaining = len(_glob.glob(str(base / "**" / "*.tif"), recursive=True))
    print(f"   ✅ MIDV-500: kept {remaining} images → {out_dir}")


# ===========================================================================
# 6. SmartDoc 2015
# ===========================================================================

def download_smartdoc15(out_dir: str) -> None:
    """Download SmartDoc 2015 and extract one optimal frame per video clip.

    The archive is downloaded via ``requests`` with a streaming read and
    parsed on-the-fly from the tarball — the 1+ GB tarfile is never
    materialised fully on disk.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/smartdoc15``).
    """
    import glob as _glob
    import pandas as pd

    base = Path(out_dir)
    img_dir = base / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    frames_url = (
        "https://github.com/jchazalon/smartdoc15-ch1-dataset/"
        "releases/download/v2.0.0/frames.tar.gz"
    )
    archive_path = base / "frames.tar.gz"

    print(f"\n📥 6/7  Downloading SmartDoc 2015 frames archive…")

    if not archive_path.exists():
        response = requests.get(frames_url, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(archive_path, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="SmartDoc download"
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))

    # Scan the tarball and extract 1 frame per video folder (frame index ≥ 50
    # avoids autofocus blur at clip start)
    print("   Parsing archive — extracting 1 optimal frame per clip…")
    df = None
    extracted_clips: dict[str, dict] = {}

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tqdm(tar, desc="SmartDoc clips"):
            name_lower = member.name.lower()

            # --- Grab embedded CSV metadata ---
            if df is None and (name_lower.endswith(".csv") or name_lower.endswith(".csv.gz")):
                fh = tar.extractfile(member)
                if fh:
                    if name_lower.endswith(".gz"):
                        with gzip.GzipFile(fileobj=fh) as gz:
                            df = pd.read_csv(gz)
                    else:
                        df = pd.read_csv(fh)

            # --- Extract first frame ≥ 50 per clip ---
            elif member.isfile() and name_lower.endswith((".png", ".jpg", ".jpeg")):
                m = re.search(r"(\d+)\.(jpeg|jpg|png)$", name_lower)
                if m and int(m.group(1)) >= 50:
                    folder = os.path.dirname(member.name)
                    clip_key = folder.replace("/", "_").replace("\\", "_").strip("_")

                    if clip_key not in extracted_clips:
                        fh = tar.extractfile(member)
                        if fh:
                            buf = np.asarray(bytearray(fh.read()), dtype=np.uint8)
                            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if img is not None:
                                out_name = f"smartdoc_{clip_key}.jpg"
                                cv2.imwrite(
                                    str(img_dir / out_name), img,
                                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                                )
                                extracted_clips[clip_key] = {
                                    "original_path": member.name,
                                    "width": img.shape[1],
                                    "height": img.shape[0],
                                }

    archive_path.unlink(missing_ok=True)

    if df is None:
        raise ValueError("Could not find the embedded CSV metadata in the archive.")

    # Map ground-truth quad coordinates from CSV to our annotation format
    annotations: dict = {}
    for clip_key, info in extracted_clips.items():
        fname = f"smartdoc_{clip_key}.jpg"
        row = df[df["image_path"] == info["original_path"]]
        if row.empty:
            row = df[df["image_path"].str.endswith(os.path.basename(info["original_path"]))]
        if not row.empty:
            r = row.iloc[0]
            annotations[fname] = {
                "width": info["width"],
                "height": info["height"],
                "corners": [
                    [r["tl_x"], r["tl_y"]],
                    [r["tr_x"], r["tr_y"]],
                    [r["br_x"], r["br_y"]],
                    [r["bl_x"], r["bl_y"]],
                ],
            }

    (base / "annotations.json").write_text(json.dumps(annotations, indent=2))
    print(f"   ✅ SmartDoc 2015: {len(annotations)} clips → {out_dir}")


# ===========================================================================
# 7. Mendeley Document Corner Dataset (validation benchmark)
# ===========================================================================

def download_mendeley(out_dir: str) -> None:
    """Download and unpack the Mendeley document corner benchmark dataset.

    This dataset is kept strictly as the **validation** partition and is
    never used during training.

    Parameters
    ----------
    out_dir : str
        Root output directory (e.g. ``raw_datasets/mendeley``).
    """
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    download_url = "https://data.mendeley.com/public-api/zip/x3nm4cxr83/download/1"
    out_zip = base / "mendeley_archive.zip"

    print(f"\n📥 7/7  Downloading Mendeley Corner dataset (~809 MB)…")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }

    with requests.get(download_url, headers=headers, stream=True,
                      allow_redirects=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_zip, "wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, desc="Mendeley download"
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))

    if not zipfile.is_zipfile(out_zip):
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP archive: {out_zip}\n"
            "Try visiting https://data.mendeley.com/datasets/x3nm4cxr83/1 manually."
        )

    print("   Extracting…")
    with zipfile.ZipFile(out_zip) as z:
        z.extractall(base)
    out_zip.unlink()

    # Mendeley zips often contain nested zip bundles — unpack them too
    for nested in base.rglob("*.zip"):
        if zipfile.is_zipfile(nested):
            print(f"   Unpacking nested archive: {nested.name}")
            with zipfile.ZipFile(nested) as nz:
                nz.extractall(base)
            nested.unlink()

    print(f"   ✅ Mendeley: extracted → {out_dir}")


# ===========================================================================
# Dispatcher
# ===========================================================================

def _run_all(
    datasets: list[str],
    out_base: str,
    hf_token: str | None,
) -> None:
    """Download each requested dataset into ``out_base/<dataset_key>/``."""
    dispatch = {
        "doc_corner":  lambda: download_doc_corner(str(Path(out_base) / "doc_corner")),
        "cord":        lambda: download_cord(str(Path(out_base) / "cord")),
        "doclaynet":   lambda: download_doclaynet(str(Path(out_base) / "doclaynet")),
        "doc3d":       lambda: download_doc3d(
                           str(Path(out_base) / "doc3d"), hf_token=hf_token
                       ),
        "midv500":     lambda: download_midv500(str(Path(out_base) / "midv500")),
        "smartdoc15":  lambda: download_smartdoc15(str(Path(out_base) / "smartdoc15")),
        "mendeley":    lambda: download_mendeley(str(Path(out_base) / "mendeley")),
    }

    for key in datasets:
        if key not in dispatch:
            print(f"⚠️  Unknown dataset key '{key}' — skipping.")
            continue
        try:
            dispatch[key]()
        except Exception as exc:
            print(f"❌  Error downloading '{key}': {exc}")
            import traceback
            traceback.print_exc()


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download all document corner detection training datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        metavar="DATASET",
        help=(
            "Which datasets to download. "
            f"Choices: {', '.join(ALL_DATASETS)}. "
            "Defaults to all."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT,
        help="Root directory for all raw dataset output.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point."""
    args = _parse_args()

    hf_token = _get_hf_token()
    _hf_login(hf_token)

    print(f"\n🗂️  Output directory : {args.out_dir}")
    print(f"📋  Datasets to fetch: {', '.join(args.datasets)}\n")

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    _run_all(args.datasets, args.out_dir, hf_token)

    print("\n" + "=" * 55)
    print("🎉 Download phase complete!")
    print(f"   Run `python data_prep/build_yolo_format.py` next.")
    print("=" * 55)


if __name__ == "__main__":
    main()

