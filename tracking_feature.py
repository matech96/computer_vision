import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import matech_utilities as mu

plt.rcParams['figure.figsize'] = (20, 15)

img_dir = "images/"
MIN_MATCH_COUNT = 8
# r, h, c, w = 450, 100, 890, 100  # rubber
# r,h,c,w = 250,300,850,75 # election day bracelet
# r,h,c,w = 200,200,800,350 # blue
# r, h, c, w = 200, 400, 525, 300  # head
# r, h, c, w = 75, 450, 825, 350  # german
r, h, c, w = 220, 340, 680, 180  # calculator
track_window = (c, r, w, h)
org_box = np.array([[c, r, 1],
                    [c, r + h, 1],
                    [c + w, r + h, 1],
                    [c + w, r, 1]
                    ])

cap = cv.VideoCapture(img_dir + "slow_calculator.mp4")
_, org_frame = cap.read()
org_frame = cv.cvtColor(org_frame, cv.COLOR_BGR2GRAY)
org_mask = mu.create_mask(org_frame)
org_mask[r:r + h, c:c + w] = 255
_, org_pts, org_feat = mu.extract_features(org_frame, resize=False, mask=org_mask)
org_pts = np.array([p.pt for p in org_pts], dtype=np.float32).reshape((-1, 1, 2))
mu.show_pics(1, 1, [mu.draw_points(org_pts.reshape((-1, 2)), org_frame)])

prev_pts = org_pts
prev_feat = org_feat
prev_box = org_box
prev_frame = org_frame

fourcc = cv.VideoWriter_fourcc(*'DIVX')
vh, vw = org_frame.shape
out = cv.VideoWriter('output2.mkv', fourcc, 25.0, (vw, vh), isColor=True)
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    ret, frame = cap.read()

    if ret:
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pts, st, err = cv.calcOpticalFlowPyrLK(org_frame, prev_frame, prev_pts, None, **lk_params)
        m_pts = pts[st == 1]
        m_prev_pts = prev_pts[st == 1]
        # h, _ = cv.findHomography(m_prev_pts, m_pts, cv.RANSAC, 5.0)
        # box = mu.transform_with_homography(h, prev_box)
        # res, center = mu.draw_box_homogeneous(box, frame, True)
        res = frame
        res = mu.draw_points(m_pts, res)
        prev_pts = m_pts.reshape(-1, 1, 2)
        # prev_box = box
        prev_frame = gray_frame
        # mu.cv_show_pics(1, 1, [res])
        out.write(res)
    else:
        break

out.release()
cap.release()
