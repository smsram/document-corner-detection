# Contributing to Document Corner Detection

Thank you for your interest in contributing! This document explains how to get started, the project conventions, and what kinds of contributions are most welcome.

---

## 🐛 Reporting Issues

Before opening a new issue, please:

1. Search [existing issues](../../issues) to see if it has already been reported.
2. If not, open a new issue with:
   - A clear title and description of the problem.
   - Minimal reproduction steps (image path, command run, Python version).
   - Full traceback if applicable.

---

## 🚀 Submitting Pull Requests

### Setup

```bash
git clone https://github.com/smsram/document-corner-detection.git
cd document-corner-detection
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Workflow

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Make your changes**, following the code style guidelines below.
3. **Test** your changes:
   ```bash
   # Quick syntax check
   python -m py_compile src/postprocess.py src/infer.py src/webcam.py src/export.py
   python -m py_compile data_prep/download_datasets.py data_prep/build_yolo_format.py
   python -m py_compile training/train.py training/test.py

   # Run inference smoke test (requires weights/best.pt)
   python src/infer.py --image assets/manual-capture.jpg --no-display
   ```
4. **Commit** with a descriptive message following [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat: add batch inference script
   fix: handle empty mask prediction in get_strict_4_corners
   docs: update dataset acknowledgements table
   ```
5. **Push** and open a Pull Request against `main`.

---

## 🎨 Code Style

- Follow **PEP 8** (max line length: 100 characters).
- All public functions and classes **must have docstrings** (NumPy style preferred).
- Inline comments are required for all geometric math operations.
- Use **type hints** (`str | None`, `np.ndarray`, etc.) on function signatures.
- Do not add `print()` calls in library code — use `logging` or return values.

---

## 🔬 What to Contribute

High-priority contributions:

| Area | Ideas |
|---|---|
| **Accuracy** | Add more datasets to `data_prep/download_datasets.py` |
| **Post-processing** | Improve `get_strict_4_corners` for circular or hexagonal document shapes |
| **Android** | Kotlin demo app using the exported `.tflite` model |
| **API** | REST inference server (`src/serve.py`) with FastAPI |
| **Metrics** | Corner-IoU score against ground-truth quads in `training/test.py` |
| **Export** | ONNX export path in `src/export.py` |

---

## 📜 Licence

By submitting a pull request you agree that your contribution will be licensed under the [MIT License](LICENSE).

