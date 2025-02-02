import cv2
import numpy as np
import utils
from imutils import contours
import imutils

img_path = "3.jpg"
answer_key = {0: 1, 1: 4, 2: 0, 3: 3, 4: 5}
out_img_width = 600
out_img_height = 600
im = cv2.imread(img_path)

im = cv2.resize(im, (out_img_width, out_img_height))
im_contours = im.copy()

img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
img_canny = cv2.Canny(img_blur, 75, 200)

contourss, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im_contours, contourss, -1, (0, 255, 0), 5)
rectCont = utils.rect_contours(contourss)
biggestcont = rectCont[0]
biggest_rect_corners = utils.get_rect_corner_points(biggestcont)

if biggest_rect_corners.size != 0:
    cv2.drawContours(im, biggest_rect_corners, -1, (0, 255, 0), 20)
    biggest_rect_corners = utils.reorder(biggest_rect_corners)

    pt1 = np.float32(biggest_rect_corners)
    pt2 = np.float32(
        [
            [0, 0],
            [out_img_width, 0],
            [0, out_img_height],
            [out_img_width, out_img_height],
        ]
    )
    transformation_matrix = cv2.getPerspectiveTransform(pt1, pt2)
    im_warped_colored = cv2.warpPerspective(
        im, transformation_matrix, (out_img_width, out_img_height)
    )
    im_warped_gray = cv2.cvtColor(im_warped_colored, cv2.COLOR_BGR2GRAY)

    thresh_img = cv2.threshold(im_warped_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspectRatio = w / float(h)
        if w >= 10 and h >= 10 and aspectRatio >= 0.9 and aspectRatio <= 1.1:
            questionCnts.append(c)

    questionCnts = contours.sort_contours(questionCnts, method="top-to-botton")[0]
    correct = 0

    for q, i in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i : i + 5])[0]
        bubbled = None

        for j, c in enumerate(cnts):
            mask = np.zeros(thresh_img.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh_img, thresh_img, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
            color = (0, 0, 255)
            k = answer_key[q]
            if k == bubbled[1]:
                correct = correct + 1

    print(correct)

    cv2.imshow("warped", thresh_img)
    cv2.waitKey(0)
