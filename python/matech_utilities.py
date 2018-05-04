from typing import *

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_pics(plot_column_size: int, plot_row_size: int, imgs: List[np.array], kargs: Dict) -> None:
    for i in range(plot_column_size * plot_row_size):
        plt.subplot(plot_row_size, plot_column_size, i + 1)
        img = imgs[i]
        d = len(img.shape)
        if d != 3:
            kargs['cmap'] = 'gray'
        plt.imshow(img, **kargs)
    plt.show()


def cv_show_pics(plot_column_size: int, plot_row_size: int, imgs: List[np.array], kargs: Dict) -> None:
    cv_imgs = []
    for img in imgs:
        cv_imgs.append(cv.cvtColor(img.copy(), cv.COLOR_BGR2RGB))
    show_pics(plot_column_size, plot_row_size, cv_imgs, **kargs)


def draw_box_homogeneous(pts: np.array, frame: np.array, draw_center: bool = False,
                         color: Tuple[int, int, int] = (255, 0, 0)) -> object:
    pts = np.array([[int(c[0] / c[2]), int(c[1] / c[2])] for c in pts])
    res = draw_poly_lines(color, frame.copy(), pts)
    if draw_center:
        center = np.mean(pts, axis=0)
        res = cv.circle(res, (int(center[0]), int(center[1])), 5, color)
        return res, center
    else:
        return res


def draw_poly_lines(color: Tuple[int, int, int], frame: np.array, pts: np.array) -> np.array:
    pts = pts.reshape((-1, 1, 2))
    res = cv.polylines(frame.copy(), [pts], True, color, thickness=5)
    return res


def draw_points(pts: np.array, frame: np.array, color: Tuple[int, int, int] = (0, 255, 255)) -> np.array:
    res = frame.copy()
    for p in pts:
        res = cv.circle(res, (int(p[0]), int(p[1])), 5, color, -1)
    return res


def resize_greatest_ax(img: np.array, s: float) -> np.array:
    y, x = img.shape[:2]
    if x > y:
        r = s / x
        nx = s
        ny = y * r
    else:
        r = s / y
        nx = x * r
        ny = s
    return cv.resize(img.copy(), (int(nx), int(ny)))


def extract_features(img: np.array, resize: bool = True, mask: np.array = None) -> Tuple[np.array, List, List]:
    img = cv.UMat(img)
    if resize:
        img = resize_greatest_ax(img, 512)

    descriptor = cv.xfeatures2d.SURF_create(400)
    (kps, features) = descriptor.detectAndCompute(img, mask)
    return img, kps, features


def create_mask_like(img: np.array) -> np.array:
    h, w = img.shape[:2]
    return create_mask(h, w)


def create_mask(h: int, w: int) -> np.array:
    return np.zeros((h, w), dtype=np.uint8)


def shape_to_homogeneous_box(shape: Tuple[int, int], upper_left_ratio: int = 4, side_ratio: int = 2) \
        -> Tuple[np.array, np.array]:
    c, h, r, vh, vw, w = shape_to_rhcw(shape, side_ratio, upper_left_ratio)
    mask, org_box = rhcw_to_homogeneous_box(r, h, c, w, vh, vw)
    return org_box, mask


def rhcw_to_homogeneous_box(r, h, c, w, vh, vw):
    org_box = np.array([[c, r, 1],
                        [c, r + h, 1],
                        [c + w, r + h, 1],
                        [c + w, r, 1]
                        ])
    mask = create_mask(vh, vw)
    mask[r:r + h, c:c + w] = 255
    return mask, org_box


def shape_to_rhcw(shape, side_ratio, upper_left_ratio):
    vh, vw = shape[:2]
    r, h, c, w = np.int32((vh / upper_left_ratio, vh / side_ratio, vw / upper_left_ratio, vw / side_ratio))
    return c, h, r, vh, vw, w


def match_features(f1: List, f2: List) -> List:
    bf = cv.DescriptorMatcher_create("BruteForce")
    matches = bf.knnMatch(f1, f2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def find_homography(pts1, pts2, matches):
    m_pts1, m_pts2 = match_points(matches, pts1, pts2)
    h, _ = cv.findHomography(m_pts1, m_pts2, cv.RANSAC, 5.0)
    return h


def match_points(matches, pts1, pts2):
    m_pts1 = np.array([pts1[m.queryIdx] for m in matches])
    m_pts2 = np.array([pts2[m.trainIdx] for m in matches])
    return m_pts1, m_pts2


def transform_with_homography(h: np.array, pts: np.array) -> np.array:
    pts = h.dot(pts.T).T
    return np.array([[int(c[0] / c[2]), int(c[1] / c[2]), 1] for c in pts])


def points_to_keypoints(pts, prev_kpts):
    kpts = []
    for pt, pkpt in zip(pts, prev_kpts):
        kpt = pkpt
        kpt.pt = (pt[0], pt[1])
        kpts.append(kpt)
    return kpts
