import cv2 as cv
import numpy as np

import matech_utilities as mu
from WebCamTracking import WebCamTracking


class TrackingObjectFlow:

    def __init__(self) -> None:
        super().__init__()

    def __startup(self, frame):
        vh, vw, _ = frame.shape
        r, h, c, w = np.int32((vh / 4, vh / 2, vw / 4, vw / 2))
        org_box = np.array([[c, r, 1],
                            [c, r + h, 1],
                            [c + w, r + h, 1],
                            [c + w, r, 1]
                            ])

        res, _ = mu.draw_box_homogeneous(org_box, frame, True)

        # Display the resulting frame
        cv.imshow('frame', res)
        if cv.waitKey(1) & 0xFF == ord('c'):
            org_mask = mu.create_mask(frame)
            org_mask[r:r + h, c:c + w] = 255
            # c. features
            self.old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, p00, _ = mu.extract_features(self.old_gray, resize=False, mask=org_mask)
            self.p0 = np.array([p.pt for p in p00], dtype=np.float32).reshape((-1, 1, 2))
            self.prev_box = org_box
            return False
        else:
            return True

    def __loop(self, frame):
        # Optic flow
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        p1, st, err = cv.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None,
                                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.00001))
        # pair points
        condition = np.array(st == 1)  # & np.array(err < 4.0)
        good_new = p1[condition]
        good_old = self.p0[condition]
        # homography
        h, m = cv.findHomography(good_old, good_new, 0, 5.0)
        h = np.around(h, decimals=10)
        box = mu.transform_with_homography(h, self.prev_box)

        res = mu.draw_points(good_new, frame)
        res, box_center = mu.draw_box_homogeneous(box, res, True)
        cv.imshow('frame', res)

        # update
        self.old_gray = frame_gray.copy()
        self.p0 = good_new.reshape(-1, 1, 2)
        self.prev_box = box

        if cv.waitKey(1) & 0xFF == ord('q'):
            return False
        else:
            return True

    def run_loops(self):
        cam = WebCamTracking(self.__startup, self.__loop)
        cam.run_loops()
        cv.destroyAllWindows()
