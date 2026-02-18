# Automated Optical Marksheet Evaluation
Experimenting with different OMR methods for their roubustness on marksheets with differnet styles.

#### *`Output of threshold and image split based method`*
In this method after extracting OMR document from the input image, answer bubbles are detected based on threshold and  
their location is detected based on splitting image into grid based on number of questions and choices
![](3_out.jpg)

#### *`Output of two staged contour method`*
![](3_2staged_out.jpg)  
![](1_2staged_out.jpg)

## *Note on methods tried*
Methods tried so far and several others researched on internet are mainly based on thresholding for answer bubble detection  
and their `location finding is based on some heuristic which only seem to work for particular types OMR sheet`

# Future Work
* [ ] Use an object detection model like YOLO/SSD etc for detecting bubbles and their location, group them based on questions
      they belong to and calculate score accordingly

# Requirements
os  
cv2  
imutils  
numpy
# References
1st method Guided by this [Youtube Tutorial](https://www.youtube.com/watch?v=0IqCOPlGBTs)  
2nd method Guided by this [Youtube Tutorial](https://www.youtube.com/watch?v=1TBLc8IrLvk)  

## AI-based OMR pipeline (new)
This repository now includes a model-ready grading module in `ai_based_omr.py` to support layout-robust OMR.

### What it does
- Accepts bubble detections as `(x_center, y_center, width, height)` from any detector (YOLO/SSD/custom).
- Groups detections into question rows using y-axis clustering, then sorts bubbles left-to-right as options.
- Uses a pluggable bubble fill classifier for `filled/unfilled` decisions with confidence.
- Applies validation states per question:
  - `valid_marked`
  - `blank`
  - `multi_marked`
  - `low_confidence`
  - `invalid_row`
- Computes final score against the answer key.

### Why this is more robust than fixed-grid split
The current threshold/grid approach (`split_boxes`) assumes a fixed number of rows/columns and stable layout.
The AI pipeline decouples **where bubbles are** (detector) from **which are marked** (classifier), so it can adapt to new sheet templates.

### Minimal usage
```python
from ai_based_omr import grade_sheet_from_detections, SimpleFillRatioClassifier, BubbleDetection

# detected bubbles from a detector model in normalized sheet coordinates
detections = [
    BubbleDetection(x_center=10, y_center=10, width=8, height=8),
    BubbleDetection(x_center=20, y_center=10, width=8, height=8),
]

classifier = SimpleFillRatioClassifier(fill_ratios=[[0.2, 0.8]], mark_threshold=0.4)
result = grade_sheet_from_detections(
    image_path="sheet.jpg",
    detections=detections,
    answer_key=[1],
    classifier=classifier,
    expected_options_per_question=2,
)
print(result.score_percent)
```
