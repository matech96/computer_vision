import time

import cv2 as cv
import numpy as np

import matech_utilities as mu

cap = cv.VideoCapture(0)
ret, frame = cap.read()
vh, vw, _ = frame.shape
r, h, c, w = np.int32((vh / 4, vh / 2, vw / 4, vw / 2))
org_box = np.array([[c, r, 1],
                    [c, r + h, 1],
                    [c + w, r + h, 1],
                    [c + w, r, 1]
                    ])

while True:
    # Capture frame-by-frame
    _, frame = cap.read()
    frame = cv.flip(frame, 1)

    res, _ = mu.draw_box_homogeneous(org_box, frame, True)

    # Display the resulting frame
    cv.imshow('frame', res)
    if cv.waitKey(1) & 0xFF == ord('c'):
        break

# c. mask
org_mask = mu.create_mask(frame)
org_mask[r:r + h, c:c + w] = 255
# c. features
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
_, p00, _ = mu.extract_features(old_gray, resize=False, mask=org_mask)
p0 = np.array([p.pt for p in p00], dtype=np.float32).reshape((-1, 1, 2))
prev_point_center = np.mean(p0.reshape((-1, 2)), axis=0)

prev_box = org_box

while True:
    _, frame = cap.read()
    frame = cv.flip(frame, 1)

    # Optic flow
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                          criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.00001))
    # pair points
    condition = np.array(st == 1)  # & np.array(err < 4.0)
    good_new = p1[condition]
    good_old = p0[condition]
    point_center = np.mean(good_new, axis=0)
    # homography
    h, m = cv.findHomography(good_old, good_new, 0, 5.0)
    h = np.around(h, decimals=10)
    box = mu.transform_with_homography(h, prev_box)

    res = mu.draw_points(good_new, frame)
    res, box_center = mu.draw_box_homogeneous(box, res, True)
    cv.imshow('frame', res)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # update
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    prev_box = box

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
