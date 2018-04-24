import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_pics(plot_colum_size, plot_row_size, imgs, **kargs):
    for i in range(plot_colum_size * plot_row_size):
        plt.subplot(plot_row_size, plot_colum_size, i + 1)
        img = imgs[i]
        d = len(img.shape)
        if d != 3:
            kargs['cmap'] = 'gray'
        plt.imshow(img, **kargs)
    plt.show()


def cv_show_pics(plot_colum_size, plot_row_size, imgs, **kargs):
    cv_imgs = []
    for img in imgs:
        cv_imgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    show_pics(plot_colum_size, plot_row_size, cv_imgs, **kargs)


def draw_box(pts, frame, draw_center=False, color=(255, 0, 0)):
    ptsa = pts.reshape((-1, 1, 2))
    res = cv.polylines(frame, [ptsa], True, color, thickness=5)
    if draw_center:
        center = np.mean(pts, axis=0)
        res = cv.circle(res, (int(center[0]), int(center[1])), 5, color)
        return res, center
    else:
        return res


def resize_greatest_ax(img, s):
    y, x = img.shape[:2]
    if x > y:
        r = s / x
        nx = s
        ny = y * r
    else:
        r = s / y
        nx = x * r
        ny = s
    return cv.resize(img, (int(nx), int(ny)))


def feature_match(img, resize=True, mask=None):
    gray = img  # cv.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if resize:
        gray = resize_greatest_ax(gray, 512)

    #     descriptor = cv.ORB_create(400)
    #     descriptor = cv.xfeatures2d.SIFT_create(400)
    descriptor = cv.xfeatures2d.SURF_create(400)
    (kps, features) = descriptor.detectAndCompute(gray, mask)
    return gray, kps, features


def create_mask(img):
    h, w = img.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)
