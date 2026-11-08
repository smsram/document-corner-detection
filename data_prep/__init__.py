"""
data_prep
=========
Dataset acquisition and YOLO-format assembly utilities.

Modules
-------
download_datasets : Stream and save all source datasets (DocCorner, Doc3D,
                    CORD-v2, DocLayNet, MIDV-500, SmartDoc 2015, Mendeley).
build_yolo_format : Parse raw datasets into a unified YOLOv8-seg label format
                    and generate synthetic edge-case training images.
"""

