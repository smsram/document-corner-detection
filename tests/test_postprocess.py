"""
Unit tests for geometric post-processing and 4-corner extraction.
"""

import numpy as np
import pytest
from src.postprocess import (
    sort_4_corners,
    get_strict_4_corners,
    perspective_warp,
    draw_document_overlay,
)


def test_sort_4_corners_canonical_order():
    """Verify that points in arbitrary order are sorted into [TL, TR, BR, BL]."""
    # Unordered rectangle points
    pts = np.array([[200, 300], [50, 50], [50, 300], [200, 50]], dtype=np.float32)
    sorted_pts = sort_4_corners(pts)

    assert sorted_pts.shape == (4, 2)
    # TL: min(x+y) = 50+50 = 100
    assert np.allclose(sorted_pts[0], [50, 50])
    # TR: min(x-y) = 200-50 = 150 vs 50-300=-250 -> BL is 50,300, TR is 200,50
    assert np.allclose(sorted_pts[1], [200, 50])
    # BR: max(x+y) = 200+300 = 500
    assert np.allclose(sorted_pts[2], [200, 300])
    # BL: max(x-y)
    assert np.allclose(sorted_pts[3], [50, 300])


def test_get_strict_4_corners_redundant_vertices():
    """Verify that redundant boundary points along polygon edges are simplified to 4 vertices."""
    # Rectangle with 8 vertices (intermediates on edges)
    poly = np.array([
        [10, 10], [50, 10], [100, 10],
        [100, 50], [100, 100],
        [50, 100], [10, 100],
        [10, 50]
    ], dtype=np.int32)

    corners = get_strict_4_corners(poly)
    assert corners.shape == (4, 2)
    assert np.allclose(corners[0], [10, 10])
    assert np.allclose(corners[1], [100, 10])
    assert np.allclose(corners[2], [100, 100])
    assert np.allclose(corners[3], [10, 100])


def test_perspective_warp_dimensions():
    """Verify that perspective_warp rectifies a quadrilateral to positive dimensions."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    corners = np.array([
        [100, 100],
        [500, 80],
        [520, 400],
        [110, 420]
    ], dtype=np.float32)

    warped = perspective_warp(img, corners)
    assert warped is not None
    assert warped.ndim == 3
    assert warped.shape[0] > 0
    assert warped.shape[1] > 0


def test_draw_document_overlay():
    """Verify that draw_document_overlay returns an image of matching shape."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    corners = np.array([[50, 50], [350, 50], [350, 350], [50, 350]], dtype=np.float32)
    annotated = draw_document_overlay(img, corners)

    assert annotated.shape == img.shape
    assert annotated.dtype == img.dtype
