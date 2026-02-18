


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


def download_file(url: str, output_path: str) -> str:
    """Download a model/checkpoint file if it does not exist locally."""

    if os.path.exists(output_path):
        return output_path

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    urllib.request.urlretrieve(url, output_path)
    return output_path


def download_default_yolo_model(output_path: str = "models/yolov8n.pt") -> str:
    """Download a default YOLO checkpoint for quick bootstrapping."""

    return download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        output_path,
    )




def _require_cv2():
    if importlib.util.find_spec("cv2") is None:
        raise RuntimeError("opencv-python is not installed. Install it to use image/model features.")
    return importlib.import_module("cv2")




def _require_numpy():
    if importlib.util.find_spec("numpy") is None:
        raise RuntimeError("numpy is not installed. Install it to use image/model features.")
    return importlib.import_module("numpy")

class YOLOBubbleDetector:
    """YOLO detector wrapper for bubble localization.

    Requires the `ultralytics` package and a trained detector model.
    """

    def __init__(
        self,
        model_path: str,
        bubble_class_name: str = "bubble",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        if importlib.util.find_spec("ultralytics") is None:
            raise RuntimeError(
                "ultralytics is not installed. Install it to use YOLOBubbleDetector."
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        ultralytics = importlib.import_module("ultralytics")
        self._model = ultralytics.YOLO(model_path)
        self._bubble_class_name = bubble_class_name
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold

    def detect(self, image_path: str) -> List[BubbleDetection]:
        cv2 = _require_cv2()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        height, width = image.shape[:2]

        results = self._model.predict(
            source=image_path,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )

        detections: List[BubbleDetection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                if cls_name != self._bubble_class_name:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    _detection_from_bbox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        confidence=float(box.conf[0].item()),
                        image_width=width,
                        image_height=height,
                    )
                )

        return detections


class WarpedGridBubbleDetector:
    """Detector that warps the sheet and emits a normalized fixed-grid bubble layout.

    This is useful for end-to-end execution from `main.py` while transitioning to
    learned detectors.
    """

    def __init__(self, num_questions: int, num_choices: int, output_size: Tuple[int, int] = (700, 700)):
        self._num_questions = num_questions
        self._num_choices = num_choices
        self._output_size = output_size

    def detect(self, image_path: str) -> List[BubbleDetection]:
        cv2 = _require_cv2()

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        out_w, out_h = self._output_size
        image = cv2.resize(image, (out_w, out_h))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        canny = cv2.Canny(blur, 10, 50)

        contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        quadrilaterals = []
        for contour in contours:
            if cv2.contourArea(contour) <= 50:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                quadrilaterals.append(contour)

        if not quadrilaterals:
            raise RuntimeError("No OMR sheet contour detected for warped-grid detector.")

        biggest = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[0]
        perimeter = cv2.arcLength(biggest, True)
        corners = cv2.approxPolyDP(biggest, 0.02 * perimeter, True)
        points = corners.reshape((4, 2))

        ordered = _reorder_points(points)
        pt1 = np_float32(ordered)
        pt2 = np_float32([[0, 0], [out_w, 0], [0, out_h], [out_w, out_h]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        _ = cv2.warpPerspective(image, matrix, (out_w, out_h))

        detections: List[BubbleDetection] = []
        box_w = 1.0 / float(self._num_choices)
        box_h = 1.0 / float(self._num_questions)
        for q_idx in range(self._num_questions):
            for c_idx in range(self._num_choices):
                detections.append(
                    BubbleDetection(
                        x_center=(c_idx + 0.5) * box_w,
                        y_center=(q_idx + 0.5) * box_h,
                        width=box_w,
                        height=box_h,
                        confidence=1.0,
                    )
                )

        return detections


class SSDBubbleDetector:
    """OpenCV DNN SSD detector wrapper for bubble localization.

    Expects an ONNX/Caffe/TensorFlow SSD-style detection network that outputs
    boxes + confidences, and uses class-id filtering for bubble class.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        bubble_class_id: int = 1,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.40,
        input_size: Tuple[int, int] = (640, 640),
        scale: float = 1.0 / 255.0,
        swap_rb: bool = True,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SSD model not found: {model_path}")
        if config_path and not os.path.isfile(config_path):
            raise FileNotFoundError(f"SSD config not found: {config_path}")

        cv2 = _require_cv2()
        self._net = cv2.dnn_DetectionModel(model_path, config_path or "")
        self._net.setInputSize(input_size[0], input_size[1])
        self._net.setInputScale(scale)
        self._net.setInputSwapRB(swap_rb)
        self._bubble_class_id = bubble_class_id
        self._confidence_threshold = confidence_threshold
        self._nms_threshold = nms_threshold

    def detect(self, image_path: str) -> List[BubbleDetection]:
        cv2 = _require_cv2()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image_height, image_width = image.shape[:2]

        class_ids, confidences, boxes = self._net.detect(
            image,
            confThreshold=self._confidence_threshold,
            nmsThreshold=self._nms_threshold,
        )

        detections: List[BubbleDetection] = []
        if len(class_ids) == 0:
            return detections

        for class_id, confidence, box in zip(class_ids.flatten(), confidences.flatten(), boxes):
            if int(class_id) != self._bubble_class_id:
                continue
            x, y, w, h = [float(v) for v in box]
            detections.append(
                _detection_from_bbox(
                    x1=x,
                    y1=y,
                    x2=x + w,
                    y2=y + h,
                    confidence=float(confidence),
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        return detections


class ONNXBubbleFillClassifier:
    """Binary fill classifier backed by an ONNX model via OpenCV DNN.

    The ONNX model should output either:
    - single sigmoid probability for filled class, OR
    - 2 logits/probabilities [empty, filled]
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (32, 32),
        mark_threshold: float = 0.50,
        normalize: bool = True,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"ONNX classifier model not found: {model_path}")
        cv2 = _require_cv2()
        self._net = cv2.dnn.readNetFromONNX(model_path)
        self._input_size = input_size
        self._mark_threshold = mark_threshold
        self._normalize = normalize

    def classify(
        self, image_path: str, detection: BubbleDetection, option_index: int, question_index: int
    ) -> BubbleDecision:
        cv2 = _require_cv2()
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        crop = _crop_detection(image, detection)
        crop = cv2.resize(crop, self._input_size, interpolation=cv2.INTER_AREA)

        scale = 1.0 / 255.0 if self._normalize else 1.0
        blob = cv2.dnn.blobFromImage(crop, scalefactor=scale, size=self._input_size)

        self._net.setInput(blob)
        output = self._net.forward().reshape(-1)
        fill_prob = _extract_fill_probability(output)

        return BubbleDecision(
            is_filled=fill_prob >= self._mark_threshold,
            fill_ratio=fill_prob,
            confidence=abs(fill_prob - self._mark_threshold) * 2,
        )


class PixelDensityBubbleFillClassifier:
    """CV baseline classifier using adaptive threshold + filled-pixel ratio."""

    def __init__(self, mark_threshold: float = 0.45):
        self._mark_threshold = mark_threshold

    def classify(
        self, image_path: str, detection: BubbleDetection, option_index: int, question_index: int
    ) -> BubbleDecision:
        cv2 = _require_cv2()
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        crop = _crop_detection(image, detection)
        binary = cv2.adaptiveThreshold(
            crop,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            5,
        )
        fill_ratio = float(cv2.countNonZero(binary)) / float(binary.shape[0] * binary.shape[1])

        return BubbleDecision(
            is_filled=fill_ratio >= self._mark_threshold,
            fill_ratio=fill_ratio,
            confidence=min(1.0, abs(fill_ratio - self._mark_threshold) * 2),
        )


class SimpleFillRatioClassifier:
    """Classifier adapter for pre-computed fill ratios.

    Useful for tests where a model is not yet available.
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


def _crop_detection(image, detection: BubbleDetection, padding_ratio: float = 0.2):
    image_height, image_width = image.shape[:2]

    width_pixels = max(1, int(round(detection.width * image_width)))
    height_pixels = max(1, int(round(detection.height * image_height)))
    center_x = int(round(detection.x_center * image_width))
    center_y = int(round(detection.y_center * image_height))

    pad_x = int(round(width_pixels * padding_ratio))
    pad_y = int(round(height_pixels * padding_ratio))

    x1 = max(0, center_x - width_pixels // 2 - pad_x)
    y1 = max(0, center_y - height_pixels // 2 - pad_y)
    x2 = min(image_width, center_x + width_pixels // 2 + pad_x)
    y2 = min(image_height, center_y + height_pixels // 2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        np = _require_numpy()
        return np.zeros((8, 8), dtype=np.uint8)
    return image[y1:y2, x1:x2]


def _reorder_points(points):
    points_sum = [float(p[0] + p[1]) for p in points]
    points_diff = [float(p[0] - p[1]) for p in points]

    ordered = [None, None, None, None]
    ordered[0] = points[points_sum.index(min(points_sum))]
    ordered[3] = points[points_sum.index(max(points_sum))]
    ordered[1] = points[points_diff.index(max(points_diff))]
    ordered[2] = points[points_diff.index(min(points_diff))]
    return ordered


def np_float32(values):
    np = _require_numpy()
    return np.float32(values)


def _extract_fill_probability(output) -> float:
    values = list(output)
    if len(values) == 1:
        value = float(values[0])
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-value))
        return float(min(1.0, max(0.0, value)))

    if len(values) >= 2:
        a = float(values[0])
        b = float(values[1])
        if max(a, b) > 1.0 or min(a, b) < 0.0:
            m = max(a, b)
            exps_a = math.exp(a - m)
            exps_b = math.exp(b - m)
            denom = exps_a + exps_b
            prob_b = exps_b / denom if denom > 0 else 0.0
            return float(min(1.0, max(0.0, prob_b)))

        total = a + b
        if total <= 0:
            return 0.0
        return float(min(1.0, max(0.0, b / total)))

    return 0.0


def _detection_from_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float,
    image_width: int,
    image_height: int,
) -> BubbleDetection:
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    center_x = x1 + box_width / 2.0
    center_y = y1 + box_height / 2.0

    x_center_norm = float(min(1.0, max(0.0, center_x / max(1, image_width))))
    y_center_norm = float(min(1.0, max(0.0, center_y / max(1, image_height))))
    width_norm = float(min(1.0, max(1e-6, box_width / max(1, image_width))))
    height_norm = float(min(1.0, max(1e-6, box_height / max(1, image_height))))

    return BubbleDetection(
        x_center=x_center_norm,
        y_center=y_center_norm,
        width=width_norm,
        height=height_norm,
        confidence=float(min(1.0, max(0.0, confidence))),
        bbox=(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
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
        y_tolerance = max(6.0 / 1000.0, median_height * 0.65)

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


def grade_sheet_from_detector(
    image_path: str,
    detector: BubbleDetector,
    answer_key: Sequence[int],
    classifier: BubbleFillClassifier,
    expected_options_per_question: Optional[int] = None,
    min_mark_threshold: float = 0.40,
    min_confidence_gap: float = 0.12,
) -> SheetResult:
    detections = detector.detect(image_path)
    return grade_sheet_from_detections(
        image_path=image_path,
        detections=detections,
        answer_key=answer_key,
        classifier=classifier,
        expected_options_per_question=expected_options_per_question,
        min_mark_threshold=min_mark_threshold,
        min_confidence_gap=min_confidence_gap,
    )


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
