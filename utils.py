import cv2
import numpy as np


def rect_contours(contours):
    rects = []
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if len(approx) == 4:
                rects.append(contour)

    rects = sorted(rects, key=cv2.contourArea, reverse=True)
    return rects


def get_rect_corner_points(contour):
    perimeter = cv2.arcLength(contour, True)
    rectCornerPoints = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    return rectCornerPoints


def reorder(points):
    points = points.reshape((4, 2))
    # we are summing the coordinates x,y and smallest one will be bottom left and max will be top right

    my_points_new = np.zeros((4, 1, 2), np.int32)

    points_sum = points.sum(1)
    my_points_new[0] = points[np.argmin(points_sum)]  # (0,0)
    my_points_new[3] = points[np.argmax(points_sum)]  # (w,h)

    points_diff = np.diff(points, axis=1)
    my_points_new[1] = points[np.argmin(points_diff)]  # (w,0)
    my_points_new[2] = points[np.argmax(points_diff)]  # (0,h)

    return my_points_new


def split_boxes(img, split_num_col=5, split_num_row=5):
    rows = np.vsplit(img, split_num_row)

    boxes = []
    for r in rows:
        cols = np.hsplit(r, split_num_col)

        for box in cols:
            boxes.append(box)

    return boxes


def show_answers(img, answers, score, answer_key, num_questions, num_choices):
    width = int(img.shape[1] / num_questions)
    heigth = int(img.shape[0] / num_choices)

    for x in range(0, num_questions):
        ans = answers[x]
        c_x = (ans * width) + width // 2
        c_y = (x * heigth) + heigth // 2
        if ans == answer_key[x]:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.circle(img, (c_x, c_y), 30, color, 5)
    return img


def get_edges(img_path):
    out_img_width = 700
    out_img_height = 700

    im = cv2.imread(img_path)
    im = cv2.resize(im, (out_img_width, out_img_height))
    im_contours = im.copy()

    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)
    return out_img_width, out_img_height, im, im_contours, img_canny
