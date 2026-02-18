import importlib.util
import os
import tempfile
import unittest

from ai_based_omr import (
    download_file,
    BubbleDetection,
    PixelDensityBubbleFillClassifier,
    SimpleFillRatioClassifier,
    _detection_from_bbox,
    _extract_fill_probability,
    grade_sheet_from_detections,
    grade_sheet_from_detector,
)


class FakeDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image_path: str):
        return self._detections


class AIBasedOMRTests(unittest.TestCase):
    def test_grading_valid_and_blank(self):
        detections = [
            BubbleDetection(x_center=0.10, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.20, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.30, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.10, y_center=0.30, width=0.08, height=0.08),
            BubbleDetection(x_center=0.20, y_center=0.30, width=0.08, height=0.08),
            BubbleDetection(x_center=0.30, y_center=0.30, width=0.08, height=0.08),
        ]
        fill_ratios = [
            [0.10, 0.78, 0.15],
            [0.12, 0.18, 0.22],
        ]
        classifier = SimpleFillRatioClassifier(fill_ratios=fill_ratios, mark_threshold=0.40)
        result = grade_sheet_from_detections(
            image_path="dummy.jpg",
            detections=detections,
            answer_key=[1, 2],
            classifier=classifier,
            expected_options_per_question=3,
        )

        self.assertEqual(result.correct_count, 1)
        self.assertAlmostEqual(result.score_percent, 50.0)
        self.assertEqual(result.answers[0].selected_option, 1)
        self.assertEqual(result.answers[0].status, "valid_marked")
        self.assertIsNone(result.answers[1].selected_option)
        self.assertEqual(result.answers[1].status, "blank")

    def test_multi_marked(self):
        detections = [
            BubbleDetection(x_center=0.10, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.20, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.30, y_center=0.10, width=0.08, height=0.08),
        ]
        fill_ratios = [[0.63, 0.61, 0.10]]
        classifier = SimpleFillRatioClassifier(fill_ratios=fill_ratios, mark_threshold=0.40)
        result = grade_sheet_from_detections(
            image_path="dummy.jpg",
            detections=detections,
            answer_key=[0],
            classifier=classifier,
            expected_options_per_question=3,
            min_confidence_gap=0.05,
        )

        self.assertEqual(result.answers[0].status, "multi_marked")
        self.assertIsNone(result.answers[0].selected_option)
        self.assertEqual(result.correct_count, 0)

    def test_invalid_row_count(self):
        detections = [
            BubbleDetection(x_center=0.10, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.20, y_center=0.10, width=0.08, height=0.08),
        ]
        fill_ratios = [[0.55, 0.20]]
        classifier = SimpleFillRatioClassifier(fill_ratios=fill_ratios, mark_threshold=0.40)

        result = grade_sheet_from_detections(
            image_path="dummy.jpg",
            detections=detections,
            answer_key=[0],
            classifier=classifier,
            expected_options_per_question=3,
        )

        self.assertEqual(result.answers[0].status, "invalid_row")
        self.assertEqual(result.correct_count, 0)

    def test_grade_sheet_from_detector_wrapper(self):
        detections = [
            BubbleDetection(x_center=0.10, y_center=0.10, width=0.08, height=0.08),
            BubbleDetection(x_center=0.20, y_center=0.10, width=0.08, height=0.08),
        ]
        detector = FakeDetector(detections)
        classifier = SimpleFillRatioClassifier(fill_ratios=[[0.1, 0.8]], mark_threshold=0.40)
        result = grade_sheet_from_detector(
            image_path="dummy.jpg",
            detector=detector,
            answer_key=[1],
            classifier=classifier,
            expected_options_per_question=2,
        )
        self.assertEqual(result.correct_count, 1)

    def test_detection_conversion_normalizes_coordinates(self):
        detection = _detection_from_bbox(
            x1=10,
            y1=20,
            x2=30,
            y2=60,
            confidence=0.8,
            image_width=100,
            image_height=200,
        )
        self.assertAlmostEqual(detection.x_center, 0.20)
        self.assertAlmostEqual(detection.y_center, 0.20)
        self.assertAlmostEqual(detection.width, 0.20)
        self.assertAlmostEqual(detection.height, 0.20)

    def test_extract_fill_probability_sigmoid_and_softmax(self):
        self.assertAlmostEqual(_extract_fill_probability([0.9]), 0.9)

        sigmoid_value = _extract_fill_probability([2.0])
        self.assertGreater(sigmoid_value, 0.85)

        softmax_value = _extract_fill_probability([0.2, 1.4])
        self.assertGreater(softmax_value, 0.7)

    def test_pixel_density_classifier(self):
        if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("numpy") is None:
            self.skipTest("opencv-python/numpy not installed")

        import cv2
        import numpy as np

        image = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(image, (35, 35), (65, 65), 255, thickness=-1)

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, image)
            classifier = PixelDensityBubbleFillClassifier(mark_threshold=0.15)
            decision = classifier.classify(
                image_path=tmp.name,
                detection=BubbleDetection(
                    x_center=0.5,
                    y_center=0.5,
                    width=0.3,
                    height=0.3,
                ),
                option_index=0,
                question_index=0,
            )

        self.assertTrue(decision.is_filled)
        self.assertGreater(decision.fill_ratio, 0.15)


    def test_download_file_uses_local_copy_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = os.path.join(tmpdir, "source.txt")
            target = os.path.join(tmpdir, "target.txt")
            with open(source, "w", encoding="utf-8") as f:
                f.write("abc")

            output = download_file(f"file://{source}", target)
            self.assertEqual(output, target)
            with open(target, "r", encoding="utf-8") as f:
                self.assertEqual(f.read(), "abc")

            # second call should be a no-op and still return same path
            output2 = download_file(f"file://{source}", target)
            self.assertEqual(output2, target)

if __name__ == "__main__":
    unittest.main()
