"""
infer.py
========
Single-image document corner detection inference script.

Runs YOLOv8-seg on one image file, extracts exactly 4 document corners via
``postprocess.get_strict_4_corners``, overlays the results, and optionally
saves the annotated image and/or a perspective-warped crop.

Usage
-----
From the repository root::

    python src/infer.py --image path/to/image.jpg

    # Save annotated output
    python src/infer.py --image path/to/image.jpg --output result.jpg

    # Also save the de-warped document crop
    python src/infer.py --image path/to/image.jpg --output result.jpg --warp

    # Custom confidence threshold and weights
    python src/infer.py --image path/to/image.jpg \\
        --weights weights/best.pt --conf 0.4

    # Suppress the display window (headless / CI)
    python src/infer.py --image path/to/image.jpg --no-display
"""

import argparse
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# Allow running directly from either the repo root or src/
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from postprocess import draw_document_overlay, get_strict_4_corners, perspective_warp

# ---------------------------------------------------------------------------
# Default paths (relative to the repository root)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = os.path.join(_ROOT_DIR, "weights", "best.pt")


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_inference(
    image_path: str,
    weights_path: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    imgsz: int = 640,
) -> dict:
    """Run YOLOv8-seg inference on a single image and extract 4 document corners.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the input image.
    weights_path : str
        Path to the ``.pt`` model checkpoint.
    conf : float
        Minimum confidence threshold for detections (0–1).
    imgsz : int
        Inference image size (must match the training resolution; default 640).

    Returns
    -------
    result : dict
        ``{
            "image"       : np.ndarray,  # original BGR image
            "corners"     : np.ndarray | None,  # (4, 2) float32, or None
            "annotated"   : np.ndarray,  # BGR image with overlay drawn
            "found"       : bool,        # True if a document was detected
        }``

    Raises
    ------
    FileNotFoundError
        If ``image_path`` or ``weights_path`` does not exist.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Place best.pt in the weights/ directory or pass --weights."
        )

    # Load model
    model = YOLO(weights_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"cv2.imread returned None for path: {image_path}")

    # Run segmentation inference with high-resolution masks
    preds = model.predict(
        source=image,
        imgsz=imgsz,
        conf=conf,
        retina_masks=True,  # Full-resolution mask output
        verbose=False,
    )[0]

    corners = None
    annotated = image.copy()

    if preds.masks is not None and len(preds.masks.xy) > 0:
        # Keep only polygons with enough vertices to describe a quadrilateral
        valid_polygons = [m for m in preds.masks.xy if len(m) >= 4]

        if valid_polygons:
            # Select the largest detected polygon (primary document)
            largest = max(
                valid_polygons,
                key=lambda m: cv2.contourArea(m.astype(np.int32)),
            )

            # Convert to OpenCV-compatible int32 point array
            pts = largest.astype(np.int32).reshape((-1, 1, 2))

            # Extract exactly 4 corners via adaptive convex-hull approximation
            corners = get_strict_4_corners(pts)

            # Draw overlay on the output image
            annotated = draw_document_overlay(image, corners)

    return {
        "image": image,
        "corners": corners,
        "annotated": annotated,
        "found": corners is not None,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8-seg single-image document corner detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--weights", "-w",
        default=DEFAULT_WEIGHTS,
        help="Path to the YOLOv8-seg .pt weights file.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to save the annotated output image (optional).",
    )
    parser.add_argument(
        "--warp",
        action="store_true",
        help="Also save the perspective-warped document crop alongside --output.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (0–1).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size in pixels.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the OpenCV display window (useful for headless environments).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for single-image inference."""
    args = _parse_args()

    print(f"🔍 Running inference on: {args.image}")
    result = run_inference(
        image_path=args.image,
        weights_path=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
    )

    if result["found"]:
        corners = result["corners"]
        print("✅ Document detected!")
        print(f"   Top-Left     : ({corners[0][0]:.1f}, {corners[0][1]:.1f})")
        print(f"   Top-Right    : ({corners[1][0]:.1f}, {corners[1][1]:.1f})")
        print(f"   Bottom-Right : ({corners[2][0]:.1f}, {corners[2][1]:.1f})")
        print(f"   Bottom-Left  : ({corners[3][0]:.1f}, {corners[3][1]:.1f})")
    else:
        print("⚠️  No document detected in the image.")

    # Save annotated output
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cv2.imwrite(args.output, result["annotated"])
        print(f"💾 Annotated image saved to: {args.output}")

        # Optionally save perspective warp
        if args.warp and result["found"]:
            warped = perspective_warp(result["image"], result["corners"])
            warp_path = os.path.splitext(args.output)[0] + "_warped.jpg"
            cv2.imwrite(warp_path, warped)
            print(f"💾 Warped document saved to: {warp_path}")

    # Display the result in an OpenCV window
    if not args.no_display:
        cv2.imshow("Document Corner Detection", result["annotated"])
        print("Press any key to close the window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

