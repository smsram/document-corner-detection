"""
postprocess.py
==============
Geometric post-processing utilities for YOLOv8-seg document segmentation masks.

The core problem this module solves:
  YOLOv8-seg outputs a dense polygon (10–200 vertices) tracing the document
  contour. For scanning applications we need exactly 4 corners so that we
  can apply a homographic / perspective warp to de-skew the document.

Pipeline
--------
1. ``get_strict_4_corners``  — Wraps the raw polygon in its convex hull and
   iteratively relaxes the ``approxPolyDP`` epsilon until exactly 4 vertices
   remain.  Falls back to the 4 extreme corners of the hull when no epsilon
   produces exactly 4 points.
2. ``sort_4_corners``        — Canonically orders the 4 points as
   [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
3. ``perspective_warp``      — Computes and applies a perspective transform
   that maps the quadrilateral to a rectified rectangle of the computed
   output size.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Corner ordering
# ---------------------------------------------------------------------------

def sort_4_corners(pts: np.ndarray) -> np.ndarray:
    """Order four 2-D points in [TL, TR, BR, BL] canonical order.

    The ordering exploits two arithmetic properties of axis-aligned corners:
    * **Top-Left**     has the smallest sum  (x + y → min)
    * **Bottom-Right** has the largest  sum  (x + y → max)
    * **Top-Right**    has the smallest diff (x - y → min)
    * **Bottom-Left**  has the largest  diff (x - y → max)

    Parameters
    ----------
    pts : np.ndarray
        Array of shape ``(N, 2)`` or ``(N, 1, 2)`` containing at least 4
        points (additional points are discarded after reshape).

    Returns
    -------
    rect : np.ndarray
        Shape ``(4, 2)``, dtype ``float32``, ordered [TL, TR, BR, BL].
    """
    # Flatten any OpenCV-style (N, 1, 2) shape to (N, 2)
    pts = pts.reshape(-1, 2).astype("float32")

    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: smallest → TL, largest → BR
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-Left
    rect[2] = pts[np.argmax(s)]   # Bottom-Right

    # Difference of coordinates: smallest → TR, largest → BL
    diff = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left

    return rect


# ---------------------------------------------------------------------------
# Strict 4-corner extraction
# ---------------------------------------------------------------------------

def get_strict_4_corners(pts: np.ndarray) -> np.ndarray:
    """Extract exactly 4 corners from an arbitrary polygon via adaptive approximation.

    Algorithm
    ---------
    1. Compute the **convex hull** of the input polygon.  This removes all
       concave indentations caused by partial occlusions or mask noise.
    2. Measure the hull perimeter (``cv2.arcLength``).
    3. Sweep the ``approxPolyDP`` epsilon factor from 0.01 to 0.10 of the
       perimeter in 90 steps (≈ 0.001 resolution).  Stop as soon as the
       approximation yields **exactly 4 vertices**.
    4. If no epsilon produces 4 vertices (rare, e.g. nearly circular masks),
       fall back to ordering the full convex hull via ``sort_4_corners``.

    Parameters
    ----------
    pts : np.ndarray
        Raw segmentation polygon from ``results.masks.xy``, converted to
        ``np.int32`` and shaped as ``(N, 1, 2)`` or ``(N, 2)``.

    Returns
    -------
    corners : np.ndarray
        Shape ``(4, 2)``, dtype ``float32``, in [TL, TR, BR, BL] order.

    Notes
    -----
    ``cv2.convexHull`` expects shape ``(N, 1, 2)`` for optimal performance;
    the function reshapes internally if needed.
    ``cv2.approxPolyDP`` uses the Ramer–Douglas–Peucker algorithm to reduce
    the vertex count while preserving the polygon's approximate shape.
    """
    # Ensure the input is in OpenCV's preferred (N, 1, 2) int32 format
    pts_cv = pts.reshape(-1, 1, 2).astype(np.int32)

    # Step 1: Convex hull — eliminates concavities and noise spikes
    hull = cv2.convexHull(pts_cv)

    # Step 2: Hull perimeter used as the reference length for epsilon scaling
    perimeter = cv2.arcLength(hull, closed=True)

    if perimeter < 1e-6:
        # Degenerate case: return the centroid repeated 4 times
        centroid = pts_cv.mean(axis=0).reshape(1, 2)
        return np.repeat(centroid, 4, axis=0).astype("float32")

    # Step 3: Sweep epsilon — find the coarsest approximation with exactly 4 pts
    for eps_factor in np.linspace(0.01, 0.10, 90):
        epsilon = eps_factor * perimeter
        # approxPolyDP: RDP simplification.  closed=True enforces a closed polygon.
        approx = cv2.approxPolyDP(hull, epsilon, closed=True)
        if len(approx) == 4:
            return sort_4_corners(approx)

    # Step 4: Fallback — order the full convex hull as if it were 4 corners
    return sort_4_corners(hull)


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def perspective_warp(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply a perspective transform to de-skew a document quadrilateral.

    Given the 4 corner points of a document in the camera image, this
    function computes the target rectangle dimensions from the longest
    detected edge lengths and maps the quadrilateral to a straight-on view.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image (H × W × 3) as returned by ``cv2.imread`` or
        ``cv2.VideoCapture.read``.
    corners : np.ndarray
        Shape ``(4, 2)``, ``float32``, in [TL, TR, BR, BL] order as
        returned by ``sort_4_corners`` / ``get_strict_4_corners``.

    Returns
    -------
    warped : np.ndarray
        Rectified document image (H_out × W_out × 3).

    Notes
    -----
    Output dimensions are computed as the **maximum** of the two opposing
    edge lengths to avoid cropping the document.  ``cv2.getPerspectiveTransform``
    requires exactly 4 source/destination point pairs.
    """
    tl, tr, br, bl = corners.astype("float32")

    # Compute output width: max of the two horizontal edge lengths
    width_top = np.linalg.norm(tr - tl)
    width_bot = np.linalg.norm(br - bl)
    out_w = int(max(width_top, width_bot))

    # Compute output height: max of the two vertical edge lengths
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    out_h = int(max(height_left, height_right))

    if out_w < 1 or out_h < 1:
        return image  # Cannot warp — return original

    # Destination rectangle corners (clockwise from TL)
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype="float32",
    )

    # Compute the 3×3 homography matrix and apply it
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, M, (out_w, out_h))
    return warped


# ---------------------------------------------------------------------------
# Overlay drawing helpers
# ---------------------------------------------------------------------------

def draw_document_overlay(
    image: np.ndarray,
    corners: np.ndarray,
    quad_color: tuple = (0, 255, 0),
    corner_color: tuple = (255, 0, 0),
    fill_alpha: float = 0.25,
    line_thickness: int = 3,
    corner_radius: int = 8,
) -> np.ndarray:
    """Draw a semi-transparent quadrangle and corner anchors on a copy of ``image``.

    Parameters
    ----------
    image : np.ndarray
        Source BGR image.
    corners : np.ndarray
        Shape ``(4, 2)`` int or float, [TL, TR, BR, BL].
    quad_color : tuple
        BGR colour of the boundary lines.
    corner_color : tuple
        BGR colour of the filled corner circles.
    fill_alpha : float
        Opacity of the polygon fill layer (0 = transparent, 1 = opaque).
    line_thickness : int
        Stroke width in pixels for the boundary lines.
    corner_radius : int
        Radius in pixels of the corner anchor circles.

    Returns
    -------
    vis : np.ndarray
        Annotated image (copy of ``image``).
    """
    vis = image.copy()
    pts = corners.astype(np.int32).reshape((-1, 1, 2))

    # Semi-transparent fill
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], quad_color)
    cv2.addWeighted(overlay, fill_alpha, vis, 1 - fill_alpha, 0, vis)

    # Solid boundary lines
    cv2.polylines(vis, [pts], isClosed=True, color=quad_color, thickness=line_thickness)

    # Corner anchors: filled circle + white ring for visibility
    for cx, cy in corners.astype(int):
        cv2.circle(vis, (cx, cy), corner_radius, corner_color, thickness=-1)
        cv2.circle(vis, (cx, cy), corner_radius + 2, (255, 255, 255), thickness=2)

    return vis

