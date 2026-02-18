from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class BubbleDetection:
    """Detected OMR bubble in normalized sheet coordinates."""

    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float = 1.0
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class BubbleDecision:
    """Fill-classifier decision for one bubble."""

    is_filled: bool
    fill_ratio: float
    confidence: float


@dataclass(frozen=True)
class QuestionResult:
    question_index: int
    selected_option: Optional[int]
    status: str
    confidence: float


@dataclass(frozen=True)
class SheetResult:
    answers: List[QuestionResult]
    correct_count: int
    total_questions: int
    score_percent: float


class BubbleDetector(Protocol):
    def detect(self, image_path: str) -> List[BubbleDetection]:
        """Return bubble detections on the sheet."""


class BubbleFillClassifier(Protocol):
    def classify(
        self, image_path: str, detection: BubbleDetection, option_index: int, question_index: int
    ) -> BubbleDecision:
        """Classify one bubble as filled/unfilled with confidence."""


class SimpleFillRatioClassifier:
    """Classifier adapter for pre-computed fill ratios.

    Useful for bootstrapping or tests where a model is not yet available.
    """

    def __init__(self, fill_ratios: Sequence[Sequence[float]], mark_threshold: float = 0.45):
        self._fill_ratios = fill_ratios
        self._mark_threshold = mark_threshold

    def classify(
        self, image_path: str, detection: BubbleDetection, option_index: int, question_index: int
    ) -> BubbleDecision:
        ratio = self._fill_ratios[question_index][option_index]
        return BubbleDecision(
            is_filled=ratio >= self._mark_threshold,
            fill_ratio=ratio,
            confidence=min(1.0, max(0.0, abs(ratio - self._mark_threshold) * 2)),
        )


def _median(values: Sequence[float], default: float) -> float:
    if not values:
        return default
    values_sorted = sorted(values)
    n = len(values_sorted)
    mid = n // 2
    if n % 2 == 1:
        return values_sorted[mid]
    return (values_sorted[mid - 1] + values_sorted[mid]) / 2.0


def group_bubbles_into_rows(
    detections: Sequence[BubbleDetection], y_tolerance: Optional[float] = None
) -> List[List[BubbleDetection]]:
    """Cluster bubbles into question rows using y-axis proximity."""

    if not detections:
        return []

    sorted_detections = sorted(detections, key=lambda d: (d.y_center, d.x_center))
    if y_tolerance is None:
        median_height = _median([d.height for d in sorted_detections], default=20.0)
        y_tolerance = max(6.0, median_height * 0.65)

    rows: List[List[BubbleDetection]] = []
    row_centers: List[float] = []
    for det in sorted_detections:
        placed = False
        for idx, center in enumerate(row_centers):
            if abs(det.y_center - center) <= y_tolerance:
                rows[idx].append(det)
                row_centers[idx] = _median([d.y_center for d in rows[idx]], center)
                placed = True
                break

        if not placed:
            rows.append([det])
            row_centers.append(det.y_center)

    for row in rows:
        row.sort(key=lambda d: d.x_center)

    rows.sort(key=lambda row: _median([d.y_center for d in row], default=0.0))
    return rows


def grade_sheet_from_detections(
    image_path: str,
    detections: Sequence[BubbleDetection],
    answer_key: Sequence[int],
    classifier: BubbleFillClassifier,
    expected_options_per_question: Optional[int] = None,
    min_mark_threshold: float = 0.40,
    min_confidence_gap: float = 0.12,
) -> SheetResult:
    """Grade a sheet by grouping bubbles into rows then selecting one marked option per row.

    Returns per-question status among: valid_marked, blank, multi_marked, low_confidence, invalid_row.
    """

    rows = group_bubbles_into_rows(detections)
    total_questions = len(answer_key)

    answers: List[QuestionResult] = []
    correct = 0

    for q_idx in range(total_questions):
        if q_idx >= len(rows):
            answers.append(
                QuestionResult(
                    question_index=q_idx,
                    selected_option=None,
                    status="invalid_row",
                    confidence=0.0,
                )
            )
            continue

        row = rows[q_idx]
        if expected_options_per_question is not None and len(row) != expected_options_per_question:
            answers.append(
                QuestionResult(
                    question_index=q_idx,
                    selected_option=None,
                    status="invalid_row",
                    confidence=0.0,
                )
            )
            continue

        decisions = [
            classifier.classify(image_path, det, option_index=idx, question_index=q_idx)
            for idx, det in enumerate(row)
        ]

        sorted_by_ratio = sorted(
            enumerate(decisions), key=lambda item: item[1].fill_ratio, reverse=True
        )
        top_idx, top_decision = sorted_by_ratio[0]
        second_ratio = sorted_by_ratio[1][1].fill_ratio if len(sorted_by_ratio) > 1 else 0.0

        strong_marks = [idx for idx, d in enumerate(decisions) if d.fill_ratio >= min_mark_threshold]

        if len(strong_marks) == 0:
            answers.append(
                QuestionResult(
                    question_index=q_idx,
                    selected_option=None,
                    status="blank",
                    confidence=1.0 - top_decision.fill_ratio,
                )
            )
            continue

        if len(strong_marks) > 1 and (top_decision.fill_ratio - second_ratio) < min_confidence_gap:
            answers.append(
                QuestionResult(
                    question_index=q_idx,
                    selected_option=None,
                    status="multi_marked",
                    confidence=top_decision.fill_ratio - second_ratio,
                )
            )
            continue

        if (top_decision.fill_ratio - second_ratio) < min_confidence_gap:
            answers.append(
                QuestionResult(
                    question_index=q_idx,
                    selected_option=top_idx,
                    status="low_confidence",
                    confidence=top_decision.fill_ratio - second_ratio,
                )
            )
            continue

        status = "valid_marked"
        if top_idx == answer_key[q_idx]:
            correct += 1

        answers.append(
            QuestionResult(
                question_index=q_idx,
                selected_option=top_idx,
                status=status,
                confidence=max(top_decision.confidence, top_decision.fill_ratio - second_ratio),
            )
        )

    score_percent = (correct / total_questions * 100.0) if total_questions else 0.0
    return SheetResult(
        answers=answers,
        correct_count=correct,
        total_questions=total_questions,
        score_percent=score_percent,
    )
