"""
webcam.py
=========
Real-time document corner detection via a local OpenCV webcam feed.

Captures frames from ``cv2.VideoCapture(0)`` (the default system camera),
runs YOLOv8-seg inference on each frame, extracts exactly 4 document corners
using ``postprocess.get_strict_4_corners``, and renders a live overlay.

Controls
--------
* Press **Q** to quit.
* Press **S** to save a screenshot of the current frame + overlay.
* Press **W** to save a perspective-warped crop of the detected document.

Usage
-----
From the repository root::

    python src/webcam.py

    # Custom camera index (e.g. second USB camera)
    python src/webcam.py --camera 1

    # Custom weights and confidence
    python src/webcam.py --weights weights/best.pt --conf 0.3

    # Smaller inference resolution for faster CPU inference
    python src/webcam.py --imgsz 416
"""

import argparse
import os
import sys
import time

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

# Corner labels displayed on the overlay (TL → TR → BR → BL)
_CORNER_LABELS = ["TL", "TR", "BR", "BL"]

# Distinct BGR colours for each corner anchor
_CORNER_COLORS = [
    (51,  51,  255),   # TL — red
    (0,   187, 255),   # TR — amber
    (255, 229, 0  ),   # BR — cyan
    (119, 255, 51 ),   # BL — green
]


# ---------------------------------------------------------------------------
# Corner label rendering
# ---------------------------------------------------------------------------

def _draw_corner_labels(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """Render per-corner coloured circles and text labels onto ``image`` in-place.

    Parameters
    ----------
    image : np.ndarray
        BGR image on which to draw (modified in place).
    corners : np.ndarray
        Shape ``(4, 2)`` int, in [TL, TR, BR, BL] order.

    Returns
    -------
    image : np.ndarray
        The same array (returned for convenience chaining).
    """
    for idx, (cx, cy) in enumerate(corners.astype(int)):
        color = _CORNER_COLORS[idx]
        label = _CORNER_LABELS[idx]

        # Filled corner dot
        cv2.circle(image, (cx, cy), 7, color, thickness=-1)
        # White ring for high-contrast readability on any background
        cv2.circle(image, (cx, cy), 9, (255, 255, 255), thickness=2)

        # Text label offset to avoid overlapping the dot
        text_x = cx + 11
        text_y = cy - 11
        cv2.putText(
            image, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return image


# ---------------------------------------------------------------------------
# HUD (heads-up display) rendering
# ---------------------------------------------------------------------------

def _draw_hud(
    image: np.ndarray,
    fps: float,
    found: bool,
    save_dir: str,
) -> None:
    """Render FPS counter, detection status, and keybinding hints onto ``image``.

    Parameters
    ----------
    image : np.ndarray
        BGR image modified in place.
    fps : float
        Current measured frames-per-second.
    found : bool
        Whether a document was detected in the current frame.
    save_dir : str
        Directory where screenshots are saved (shown in HUD).
    """
    h, w = image.shape[:2]

    # --- Status pill (top-left) ---
    status_text = "Tracking Locked" if found else "Searching..."
    status_color = (51, 200, 51) if found else (0, 165, 255)   # Green / Orange
    cv2.rectangle(image, (8, 8), (200, 36), (0, 0, 0), -1)
    cv2.putText(
        image, status_text, (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA,
    )

    # --- FPS meter (top-right) ---
    fps_text = f"{fps:.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image, (w - tw - 18, 8), (w - 8, 36), (0, 0, 0), -1)
    cv2.putText(
        image, fps_text, (w - tw - 12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 255), 2, cv2.LINE_AA,
    )

    # --- Keybinding hint (bottom-left) ---
    hints = "[Q] Quit  [S] Screenshot  [W] Save Warp"
    cv2.rectangle(image, (8, h - 36), (len(hints) * 8 + 16, h - 8), (0, 0, 0), -1)
    cv2.putText(
        image, hints, (14, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_webcam(
    camera_index: int = 0,
    weights_path: str = DEFAULT_WEIGHTS,
    conf: float = 0.25,
    imgsz: int = 640,
    save_dir: str = "captures",
) -> None:
    """Launch the live webcam inference loop.

    Opens the system camera at ``camera_index``, runs YOLOv8-seg inference on
    each captured frame, and displays a real-time annotated feed.

    Parameters
    ----------
    camera_index : int
        OpenCV camera device index (0 = default/built-in camera).
    weights_path : str
        Path to the YOLOv8-seg ``.pt`` checkpoint.
    conf : float
        Minimum detection confidence threshold (0–1).
    imgsz : int
        Inference resolution in pixels (square).  Lower values run faster on
        CPU at the cost of detection accuracy on small documents.
    save_dir : str
        Directory to store ``[S]``-triggered screenshots and ``[W]``-triggered
        warp crops.

    Raises
    ------
    FileNotFoundError
        If ``weights_path`` does not exist.
    RuntimeError
        If the camera cannot be opened at the given index.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}\n"
            "Place best.pt in the weights/ directory or pass --weights."
        )

    os.makedirs(save_dir, exist_ok=True)

    print(f"📦 Loading model from: {weights_path}")
    model = YOLO(weights_path)

    print(f"📷 Opening camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera at index {camera_index}. "
            "Check that the camera is connected and not in use by another application."
        )

    # Request a standard HD resolution from the camera driver
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("✅ Webcam running. Press [Q] to quit, [S] to screenshot, [W] to save warp.")

    # -------------------------------------------------------------------
    # FPS tracking
    # -------------------------------------------------------------------
    prev_time = time.perf_counter()
    fps = 0.0

    # Cache the most recently detected corners for the HUD even on missed frames
    last_corners: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    screenshot_idx = 0

    # -------------------------------------------------------------------
    # Main inference loop
    # -------------------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to grab frame from camera — retrying...")
            time.sleep(0.05)
            continue

        last_frame = frame.copy()

        # --- YOLOv8-seg inference ---
        preds = model.predict(
            source=frame,
            imgsz=imgsz,
            conf=conf,
            retina_masks=True,
            verbose=False,
        )[0]

        corners: np.ndarray | None = None
        display = frame.copy()

        if preds.masks is not None and len(preds.masks.xy) > 0:
            # Filter to polygons with sufficient vertices
            valid = [m for m in preds.masks.xy if len(m) >= 4]

            if valid:
                # Select the dominant (largest-area) document segment
                largest = max(
                    valid,
                    key=lambda m: cv2.contourArea(m.astype(np.int32)),
                )
                pts = largest.astype(np.int32).reshape((-1, 1, 2))

                # Adaptive epsilon → exactly 4 corners
                corners = get_strict_4_corners(pts)
                last_corners = corners

                # Semi-transparent quadrangle fill + boundary
                display = draw_document_overlay(display, corners)

                # Per-corner coloured anchors + labels
                _draw_corner_labels(display, corners)

        # --- FPS calculation ---
        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-9)
        prev_time = now

        # --- HUD overlay ---
        _draw_hud(display, fps, corners is not None, save_dir)

        # --- Show frame ---
        cv2.imshow("Document Corner Detection — Webcam", display)

        # --- Keyboard handling ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:   # Q or Esc → quit
            print("👋 Exiting.")
            break

        elif key == ord("s"):              # S → screenshot
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(save_dir, f"screenshot_{ts}_{screenshot_idx:03d}.jpg")
            cv2.imwrite(path, display)
            screenshot_idx += 1
            print(f"💾 Screenshot saved: {path}")

        elif key == ord("w"):              # W → perspective warp
            if last_corners is not None and last_frame is not None:
                warped = perspective_warp(last_frame, last_corners)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(save_dir, f"warp_{ts}_{screenshot_idx:03d}.jpg")
                cv2.imwrite(path, warped)
                screenshot_idx += 1
                print(f"💾 Warped document saved: {path}")
            else:
                print("⚠️  No document detected — cannot save warp.")

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8-seg live webcam document corner detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="OpenCV camera device index.",
    )
    parser.add_argument(
        "--weights", "-w",
        default=DEFAULT_WEIGHTS,
        help="Path to YOLOv8-seg .pt weights file.",
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
        help="Inference image size (pixels). Use 416 for faster CPU inference.",
    )
    parser.add_argument(
        "--save-dir",
        default=os.path.join(_ROOT_DIR, "captures"),
        help="Directory to store screenshots and warp crops.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry-point for the live webcam session."""
    args = _parse_args()
    run_webcam(
        camera_index=args.camera,
        weights_path=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()

