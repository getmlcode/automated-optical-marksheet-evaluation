import cv2
import numpy as np
import os
import utils


def get_score(img_path, num_questions, num_choices, answer_key):
    if not (num_questions == len(answer_key)):
        print("number of questions and answer key size don't match")
        exit(0)
    if not os.path.exists(img_path):
        print("image path doesn't exist")
        exit(0)
    if not os.path.isfile(img_path):
        print("provide valid image path")
        exit(0)

    out_img_width = 700
    out_img_height = 700

    im = cv2.imread(img_path)
    im = cv2.resize(im, (out_img_width, out_img_height))
    im_contours = im.copy()

    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)

    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im_contours, contours, -1, (0, 255, 0), 5)
    rectCont = utils.rect_contours(contours)
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

        # Threshold
        im_warped_gray = cv2.cvtColor(im_warped_colored, cv2.COLOR_BGR2GRAY)
        im_thresh = cv2.threshold(im_warped_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = utils.split_boxes(im_thresh)

        num_non_zero_pix_values = np.zeros((num_questions, num_choices))
        count_col = 0
        count_row = 0
        for image in boxes:
            non_zero_pixels = cv2.countNonZero(image)
            # total_pixels = image.shape[0] * image.shape[1]
            num_non_zero_pix_values[count_row][count_col] = non_zero_pixels

            count_col += 1
            if count_col == num_choices:
                count_row += 1
                count_col = 0

        answers = []
        for x in range(0, num_questions):
            arr = num_non_zero_pix_values[x]
            max_index_val = np.where(arr == np.amax(arr))
            answers.append(max_index_val[0][0])

        correct = 0
        for x in range(0, num_questions):
            if answers[x] == answer_key[x]:
                correct += 1
        score = (correct / num_questions) * 100

        return score


def get_graded_omr(
    img_path, out_img_width, out_img_height, num_questions, num_choices, answer_key
):
    im = cv2.imread(img_path)

    im = cv2.resize(im, (out_img_width, out_img_height))
    im_contours = im.copy()

    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)

    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im_contours, contours, -1, (0, 255, 0), 5)
    rectCont = utils.rect_contours(contours)
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

        # Threshold
        im_warped_gray = cv2.cvtColor(im_warped_colored, cv2.COLOR_BGR2GRAY)
        im_thresh = cv2.threshold(im_warped_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]

        boxes = utils.split_boxes(im_thresh)

        num_non_zero_pix_values = np.zeros((num_questions, num_choices))
        count_col = 0
        count_row = 0
        for image in boxes:
            non_zero_pixels = cv2.countNonZero(image)
            # total_pixels = image.shape[0] * image.shape[1]
            num_non_zero_pix_values[count_row][count_col] = non_zero_pixels

            count_col += 1
            if count_col == num_choices:
                count_row += 1
                count_col = 0

        answers = []
        for x in range(0, num_questions):
            arr = num_non_zero_pix_values[x]
            max_index_val = np.where(arr == np.amax(arr))
            answers.append(max_index_val[0][0])

        correct = 0
        for x in range(0, num_questions):
            if answers[x] == answer_key[x]:
                correct += 1
        score = (correct / num_questions) * 100
        # Displaying answers

        graded_img = utils.show_answers(
            im_warped_colored, answers, score, answer_key, num_questions, num_choices
        )

        img_raw = np.zeros_like(im_warped_colored)
        img_raw = utils.show_answers(
            img_raw, answers, score, answer_key, num_questions, num_choices
        )

        # inverse perspective
        inv_transformation_matrix = cv2.getPerspectiveTransform(pt2, pt1)
        im_inv_warped = cv2.warpPerspective(
            img_raw, inv_transformation_matrix, (out_img_width, out_img_height)
        )

        img_final = cv2.addWeighted(im, 1, im_inv_warped, 1, 0)

        return img_final


if __name__ == "__main__":

    img_final = get_graded_omr("3.jpg", 700, 700, 5, 5, [1, 2, 0, 1, 4])
    win_name = "omr sheet"
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 500, 50)
    cv2.imshow(win_name, img_final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
