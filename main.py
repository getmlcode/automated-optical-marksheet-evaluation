import threshold_based_omr as omrt
import cv2


if __name__ == "__main__":
    img_path = "3.jpg"
    im = cv2.imread(img_path)
    im = cv2.resize(im, (700, 700))
    cv2.imshow("input omr", im)

    score = omrt.get_score(img_path, 5, 5, [1, 2, 0, 1, 4])
    img_final = omrt.get_graded_omr(img_path, 700, 700, 5, 5, [1, 2, 0, 1, 4])
    cv2.putText(
        img_final,
        str(score) + "%",
        (400, 500),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    win_name = "omr sheet"
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, 500, 50)
    cv2.imshow(win_name, img_final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
