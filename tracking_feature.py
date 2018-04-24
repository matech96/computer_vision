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
# r, h, c, w = 75, 450, 825, 350  # german
r, h, c, w = 220, 340, 680, 180  # calculator
track_window = (c, r, w, h)
org_box = np.array([[c, r, 1],
                    [c, r + h, 1],
                    [c + w, r + h, 1],
                    [c + w, r, 1]
                    ])

cap = cv.VideoCapture(img_dir + "slow_calculator.mp4")
_, first_frame = cap.read()
org_mask = mu.create_mask(first_frame)
org_mask[r:r + h, c:c + w] = 255
_, org_pts, org_feat = mu.extract_features(first_frame, resize=False, mask=org_mask)
# mu.cv_show_pics(1, 1, [mu.draw_points([m.pt for m in org_pts], first_frame)])
prev_center = np.mean(org_box[:, :-1], axis=0)
prev_pts = org_pts
prev_feat = org_feat
prev_box = org_box

fourcc = cv.VideoWriter_fourcc(*'DIVX')
vh, vw, _ = first_frame.shape
out = cv.VideoWriter('output.mkv', fourcc, 25.0, (vw, vh), isColor=True)

while True:
    ret, frame = cap.read()

    if ret:
        # print(type(prev_feat), type(prev_pts))
        # print(prev_feat.shape, len(prev_pts))
        _, pts, feat = mu.extract_features(frame, resize=False)
        matches = mu.match_features(prev_feat, feat)
        if len(matches) > MIN_MATCH_COUNT:
            m_prev_pts, m_pts = mu.match_points(matches, prev_pts, pts)
            h_m_prev_pts = np.float32([m.pt for m in m_prev_pts]).reshape(-1, 1, 2)
            h_m_pts = np.float32([m.pt for m in m_pts]).reshape(-1, 1, 2)
            h, _ = cv.findHomography(h_m_prev_pts, h_m_pts, cv.RANSAC, 5.0)
            box = mu.transform_with_homography(h, prev_box)
            # mask[:, :] = 0
            #             ptsa = cs.reshape((-1,1,2))
            # mask = cv.fillPoly(mask, [cs], 1)
            # org_feat = feat
            # org_pts = pts
            res, center = mu.draw_box_homogeneous(box, frame, True)
            res = mu.draw_points(np.array([m.pt for m in m_prev_pts]), res)
            prev_pts = m_pts
            prev_feat = np.array([feat[m.trainIdx] for m in matches])
            prev_box = box
            # speed = center - prev_center
            #             cv_show_pics(frame)
            #             print(cs)
            # mu.cv_show_pics(1, 1, [res])
            out.write(res)
            # prev_center = center
        else:
            print("Not enough matches are found - {}/{}".format(len(matches), MIN_MATCH_COUNT))

    else:
        break
out.release()

cap.release()
