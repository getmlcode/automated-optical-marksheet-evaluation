"""End-to-end runner for AI OMR pipeline.

Usage:
  python scripts/run_e2e.py --image 3.jpg --answer-key 1,2,0,1,4 --mode warped_grid
  python scripts/run_e2e.py --image 3.jpg --answer-key 1,2,0,1,4 --mode yolo --auto-download-yolo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_based_omr import (
    ONNXBubbleFillClassifier,
    PixelDensityBubbleFillClassifier,
    SSDBubbleDetector,
    WarpedGridBubbleDetector,
    YOLOBubbleDetector,
    download_default_yolo_model,
    grade_sheet_from_detector,
)


def _parse_answer_key(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _build(args, num_questions: int):
    if args.mode == "yolo":
        model_path = args.yolo_model
        if args.auto_download_yolo:
            model_path = download_default_yolo_model(model_path)
        detector = YOLOBubbleDetector(
            model_path=model_path,
            bubble_class_name=args.bubble_class,
            confidence_threshold=args.det_conf,
        )
    elif args.mode == "ssd":
        detector = SSDBubbleDetector(
            model_path=args.ssd_model,
            config_path=args.ssd_config,
            bubble_class_id=args.ssd_class_id,
            confidence_threshold=args.det_conf,
        )
    else:
        detector = WarpedGridBubbleDetector(num_questions=num_questions, num_choices=args.num_choices)

    if args.fill_onnx:
        classifier = ONNXBubbleFillClassifier(model_path=args.fill_onnx, mark_threshold=args.fill_threshold)
    else:
        classifier = PixelDensityBubbleFillClassifier(mark_threshold=args.fill_threshold)

    return detector, classifier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--answer-key", required=True, help="comma separated, e.g. 1,2,0,1,4")
    parser.add_argument("--num-choices", type=int, default=5)

    parser.add_argument("--mode", choices=["warped_grid", "yolo", "ssd"], default="warped_grid")
    parser.add_argument("--det-conf", type=float, default=0.25)

    parser.add_argument("--yolo-model", default="models/yolov8n.pt")
    parser.add_argument("--auto-download-yolo", action="store_true")
    parser.add_argument("--bubble-class", default="bubble")

    parser.add_argument("--ssd-model", default="models/ssd_bubble.onnx")
    parser.add_argument("--ssd-config", default=None)
    parser.add_argument("--ssd-class-id", type=int, default=1)

    parser.add_argument("--fill-onnx", default=None)
    parser.add_argument("--fill-threshold", type=float, default=0.45)

    args = parser.parse_args()

    answer_key = _parse_answer_key(args.answer_key)
    detector, classifier = _build(args, num_questions=len(answer_key))

    try:
        result = grade_sheet_from_detector(
            image_path=args.image,
            detector=detector,
            answer_key=answer_key,
            classifier=classifier,
            expected_options_per_question=args.num_choices,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc), "mode": args.mode, "image": os.path.basename(args.image)}, indent=2))
        return 1

    print(json.dumps(
        {
            "mode": args.mode,
            "image": os.path.basename(args.image),
            "score_percent": result.score_percent,
            "correct_count": result.correct_count,
            "total_questions": result.total_questions,
            "answers": [
                {
                    "question_index": q.question_index,
                    "selected_option": q.selected_option,
                    "status": q.status,
                    "confidence": q.confidence,
                }
                for q in result.answers
            ],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
