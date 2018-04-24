import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

import matech_utilities as mu

plt.rcParams['figure.figsize'] = (20, 15)

img_dir = "images/"
MIN_MATCH_COUNT = 10
# r, h, c, w = 450, 100, 890, 100  # rubber
# r,h,c,w = 250,300,850,75 # election day bracelet
# r,h,c,w = 200,200,800,350 # blue
# r, h, c, w = 200, 400, 525, 300  # head
r, h, c, w = 75, 450, 825, 350  # german
track_window = (c, r, w, h)
corner_points = np.array([[c, r, 1],
                          [c, r + h, 1],
                          [c + w, r + h, 1],
                          [c + w, r, 1]
                          ])

cap = cv.VideoCapture(img_dir + "german.mp4")
_, first_frame = cap.read()
mask = mu.create_mask(first_frame)
mask[r:r + h, c:c + w] = 255
_, org_pts, org_feat = mu.feature_match(first_frame, resize=False, mask=mask)
prev_center = np.mean(corner_points[:, :-1], axis=0)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
vh, vw, _ = first_frame.shape
out = cv.VideoWriter('output.mkv', fourcc, 25.0, (vw, vh), isColor=True)

while True:
    ret, frame = cap.read()

    if ret:
        _, pts, feat = mu.feature_match(frame, False)
        matches = mu.match_features(org_feat, feat)
        if len(matches) > MIN_MATCH_COUNT:
            m_pts = np.float32([pts[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            org_m_pts = np.float32([org_pts[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, mask = cv.findHomography(org_m_pts, m_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            t_c = H.dot(corner_points.T)
            cs = np.array([[int(c[0] / c[2]), int(c[1] / c[2])] for c in t_c.T])
            mask[:, :] = 0
            #             ptsa = cs.reshape((-1,1,2))
            mask = cv.fillPoly(mask, [cs], 1)
            # org_feat = feat
            # org_pts = pts
            res, center = mu.draw_box(cs, frame, True)
            speed = center - prev_center
            #             cv_show_pics(frame)
            #             print(cs)
            out.write(res)
            prev_center = center
        else:
            print("Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))
            matchesMask = None

    else:
        break
out.release()

cap.release()
