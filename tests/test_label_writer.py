"""
Unit tests for YOLO segmentation label generation.
"""

from pathlib import Path
from data_prep.build_yolo_format import write_yolo_label


def test_write_yolo_label_normalized(tmp_path: Path):
    """Verify write_yolo_label with normalized coordinates."""
    lbl_file = tmp_path / "test.txt"
    coords = [[0.1, 0.2], [0.9, 0.2], [0.8, 0.9], [0.15, 0.85]]

    write_yolo_label(lbl_file, coords, is_normalised=True)
    assert lbl_file.exists()

    content = lbl_file.read_text().strip().split()
    assert content[0] == "0"
    assert len(content) == 9  # class_id + 8 coordinate floats
    assert float(content[1]) == pytest_approx(0.1)
    assert float(content[2]) == pytest_approx(0.2)


def test_write_yolo_label_pixel_coords(tmp_path: Path):
    """Verify write_yolo_label converts pixel coordinates to [0, 1]."""
    lbl_file = tmp_path / "test_pixels.txt"
    coords = [[50, 100], [450, 100], [450, 900], [50, 900]]

    write_yolo_label(lbl_file, coords, img_w=500, img_h=1000, is_normalised=False)
    assert lbl_file.exists()

    content = lbl_file.read_text().strip().split()
    assert content[0] == "0"
    assert len(content) == 9
    assert float(content[1]) == pytest_approx(0.10)
    assert float(content[2]) == pytest_approx(0.10)
    assert float(content[3]) == pytest_approx(0.90)
    assert float(content[4]) == pytest_approx(0.10)


def pytest_approx(val, tol=1e-4):
    import pytest
    return pytest.approx(val, abs=tol)

