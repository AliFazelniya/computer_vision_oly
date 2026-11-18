# 📚 Documentation & Technical References

This directory contains technical reference documents, architectural study notes, and presentation slides used by **Team Chati Pati** during competition preparation.

---

## 📑 Contents

| File | Type | Description | Key Topics |
| :--- | :--- | :--- | :--- |
| **`OCR_Project_Study_Note.pdf`** | Study Notes | Comprehensive study notes on end-to-end OCR systems | Persian Unicode normalisation, character shaping, Raqm rendering, degradation simulation, Tesseract engine integration, character error rate (CER) metrics |
| **`yolo.pdf`** | Technical Reference | Deep dive into the evolution and mechanics of the YOLO family | Single-stage object detection, anchor vs. anchor-free paradigms, CSPDarknet backbones, PANet/FPN necks, task-aligned assigners, loss functions (CIoU/DFL) |
| **`yolo.pptx`** | Presentation | Team presentation slides on modern real-time object detection | Architecture breakdowns (YOLOv8, YOLO11), transfer learning strategies, hyperparameter tuning, model quantization & deployment |

---

## 🎯 Usage & Audience

These materials serve as foundational reading for team members and external researchers reviewing our methodology:
- **`OCR_Project_Study_Note.pdf`** directly supports the implementations found in [`ocr/phase4_persian_ocr/`](../ocr/phase4_persian_ocr/) and [`ocr/sheet.ipynb`](../ocr/sheet.ipynb).
- **`yolo.pdf`** and **`yolo.pptx`** provide theoretical context for the detection and tracking pipelines in [`object_detection/`](../object_detection/) and [`object_tracking/`](../object_tracking/).
