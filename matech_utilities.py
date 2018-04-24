import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_pics(plot_column_size, plot_row_size, imgs, **kargs):
    for i in range(plot_column_size * plot_row_size):
        plt.subplot(plot_row_size, plot_column_size, i + 1)
        img = imgs[i]
        d = len(img.shape)
        if d != 3:
            kargs['cmap'] = 'gray'
        plt.imshow(img, **kargs)
    plt.show()


def cv_show_pics(plot_column_size, plot_row_size, imgs, **kargs):
    cv_imgs = []
    for img in imgs:
        cv_imgs.append(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    show_pics(plot_column_size, plot_row_size, cv_imgs, **kargs)


def draw_box(pts, frame, draw_center=False, color=(255, 0, 0)):
    res = draw_poly_lines(color, frame, pts)
    if draw_center:
        center = np.mean(pts, axis=0)
        res = cv.circle(res, (int(center[0]), int(center[1])), 5, color)
        return res, center
    else:
        return res


def draw_poly_lines(color, frame, pts):
    pts = pts.reshape((-1, 1, 2))
    res = cv.polylines(frame, [pts], True, color, thickness=5)
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
    if resize:
        img = resize_greatest_ax(img, 512)

    descriptor = cv.xfeatures2d.SURF_create(400)
    (kps, features) = descriptor.detectAndCompute(img, mask)
    return img, kps, features


def create_mask(img):
    h, w = img.shape[:2]
    return np.zeros((h, w), dtype=np.uint8)


def match_features(f1, f2):
    bf = cv.DescriptorMatcher_create("BruteForce")
    matches = bf.knnMatch(f1, f2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def find_homography(pts1, pts2, matches):
    m_pts1 = np.float32([pts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    m_pts2 = np.float32([pts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    h, _ = cv.findHomography(m_pts1, m_pts2, cv.RANSAC, 5.0)
    return h


def transform_with_homography(h, pts):
    t_c = h.dot(pts.T)
    return np.array([[int(c[0] / c[2]), int(c[1] / c[2])] for c in t_c.T])
