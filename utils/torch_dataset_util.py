import cv2


def rgb2bgr(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)


def bgr2rgb(cv_img):
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
