# competition-prep

Reusable computer-vision toolbox extracted from the Phase 5 notebook. Every `.py` file runs on its own and executes a self-test:

    python geometry/lines.py          # -> OK: geometry/lines.py
    python run_all_tests.py           # runs every module in its own process

| Folder | Contents |
|---|---|
| cv_basics | image/video I/O, drawing, timing, logging |
| detection | bbox/IoU/NMS, detectors (blob, YOLO), classical text-region detector |
| tracking | Kalman tracker, line-crossing counter |
| geometry | line geometry (normal form, signed distance, clipping, TLS fit) |
| ocr | Persian normalisation, CER, safe Tesseract call, auto preprocessing |
| evaluation | wall-edge / MOT / detection metrics, report helpers |
| synthetic_data | wall scenes, crossing videos, Persian text scenes (deterministic by seed) |
| mock_contests | 60-minute contest runner and starter template |

Optional dependencies (ultralytics, pytesseract + Tesseract `fas`) are imported lazily; their self-tests skip if missing.
