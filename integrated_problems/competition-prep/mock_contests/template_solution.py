# Contest starter template. Copy this file at the start of a contest.
# 
# 60-minute protocol (Day 33):  0-5 read the task & write the metric  |  5-15 harness + baseline  |  15-40 fix the biggest failure, ONE change at a time
# 40-52 held-out test + fallbacks      |  52-58 freeze & run everything  |  58-60 check the output format.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from cv_basics.image_io import load_image
from cv_basics.timing import Timer, ContestClock
from cv_basics.logging_utils import get_logger
from detection.bbox import box_iou, iou_matrix, nms
from geometry.lines import line_abc, signed_distance, point_line_distance, angle_diff_deg
from ocr.normalize import normalize_persian, normalize_eval
from ocr.metrics import cer
from evaluation.metrics import score_wall, mot_eval

log = get_logger("contest")


def solve(img):
    "TODO: return your answer for ONE input. Never raise; return None if unsure (the runner supplies a default)."
    return None


if __name__ == "__main__":
    from mock_contests.contest_runner import run_wall_contest
    clock = ContestClock(60).start()
    print(run_wall_contest(solve, name="my first try").to_string(index=False))
    clock.status() if hasattr(clock, "status") else None
