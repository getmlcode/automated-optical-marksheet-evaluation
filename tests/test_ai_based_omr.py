import unittest

from ai_based_omr import BubbleDetection, SimpleFillRatioClassifier, grade_sheet_from_detections


class AIBasedOMRTests(unittest.TestCase):
    def test_grading_valid_and_blank(self):
        detections = [
            BubbleDetection(x_center=10, y_center=10, width=8, height=8),
            BubbleDetection(x_center=20, y_center=10, width=8, height=8),
            BubbleDetection(x_center=30, y_center=10, width=8, height=8),
            BubbleDetection(x_center=10, y_center=30, width=8, height=8),
            BubbleDetection(x_center=20, y_center=30, width=8, height=8),
            BubbleDetection(x_center=30, y_center=30, width=8, height=8),
        ]
        fill_ratios = [
            [0.10, 0.78, 0.15],  # q0 -> option 1 marked
            [0.12, 0.18, 0.22],  # q1 -> blank
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
            BubbleDetection(x_center=10, y_center=10, width=8, height=8),
            BubbleDetection(x_center=20, y_center=10, width=8, height=8),
            BubbleDetection(x_center=30, y_center=10, width=8, height=8),
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
            BubbleDetection(x_center=10, y_center=10, width=8, height=8),
            BubbleDetection(x_center=20, y_center=10, width=8, height=8),
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


if __name__ == "__main__":
    unittest.main()
