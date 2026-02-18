import os

from ai_based_omr import (
    ONNXBubbleFillClassifier,
    PixelDensityBubbleFillClassifier,
    SSDBubbleDetector,
    WarpedGridBubbleDetector,
    YOLOBubbleDetector,
    grade_sheet_from_detector,
)


def build_detector_and_classifier(num_questions, num_choices):
    """Build detector/classifier from env config."""

    mode = os.getenv("OMR_DETECTOR_MODE", "warped_grid").strip().lower()
    if mode == "yolo":
        yolo_model = os.getenv("OMR_YOLO_MODEL", "models/bubble_detector.pt")
        bubble_class = os.getenv("OMR_BUBBLE_CLASS", "bubble")
        detector = YOLOBubbleDetector(model_path=yolo_model, bubble_class_name=bubble_class)
    elif mode == "ssd":
        ssd_model = os.getenv("OMR_SSD_MODEL", "models/ssd_bubble.onnx")
        ssd_cfg = os.getenv("OMR_SSD_CONFIG")
        detector = SSDBubbleDetector(model_path=ssd_model, config_path=ssd_cfg)
    else:
        detector = WarpedGridBubbleDetector(num_questions=num_questions, num_choices=num_choices)

    fill_onnx = os.getenv("OMR_FILL_ONNX")
    if fill_onnx:
        classifier = ONNXBubbleFillClassifier(model_path=fill_onnx)
    else:
        classifier = PixelDensityBubbleFillClassifier(mark_threshold=0.45)

    return detector, classifier


if __name__ == "__main__":
    try:
        import cv2
        import threshold_based_omr as omrt
    except ModuleNotFoundError as err:
        raise SystemExit(
            "Missing dependency. Install OpenCV (cv2) to run main.py end-to-end."
        ) from err

    img_path = "3.jpg"
    num_questions = 5
    num_choices = 5
    answer_key = [1, 2, 0, 1, 4]

    detector, classifier = build_detector_and_classifier(num_questions, num_choices)

    ai_result = grade_sheet_from_detector(
        image_path=img_path,
        detector=detector,
        answer_key=answer_key,
        classifier=classifier,
        expected_options_per_question=num_choices,
        min_mark_threshold=0.40,
        min_confidence_gap=0.10,
    )

    legacy_score = omrt.get_score(img_path, num_questions, num_choices, answer_key)
    img_final = omrt.get_graded_omr(img_path, 700, 700, num_questions, num_choices, answer_key)

    score_text = f"AI {ai_result.score_percent:.1f}% | Legacy {legacy_score:.1f}%"
    cv2.putText(
        img_final,
        score_text,
        (60, 650),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    print("AI pipeline score:", ai_result.score_percent)
    for q in ai_result.answers:
        print(f"Q{q.question_index + 1}: status={q.status}, selected={q.selected_option}")

    win_name = "omr sheet"
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 500, 50)
    cv2.imshow(win_name, img_final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
