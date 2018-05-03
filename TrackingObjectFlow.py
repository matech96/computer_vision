import cv2 as cv
import numpy as np

import matech_utilities as mu
from TrackingFromVideo import TrackingFromVideo


class TrackingObjectFlow:
    sigma = 51

    def __init__(self) -> None:
        super().__init__()

    def __startup(self, frame):
        self.org_box, mask = mu.shape_to_homogeneous_box(frame.shape)

        res, _ = mu.draw_box_homogeneous(self.org_box, frame, True)

        # Display the resulting frame
        cv.imshow('frame', res)
        if cv.waitKey(1) & 0xFF == ord('c'):
            # c. features
            self.prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, self.org_kpts, self.org_feat = mu.extract_features(self.prev_gray, resize=False, mask=mask)
            # self.org_pts = np.array([p.pt for p in self.org_kpts], dtype=np.float32).reshape((-1, 1, 2))
            self.prev_kpts = self.org_kpts
            return False
        else:
            return True

    def __loop(self, frame):
        # Optic flow
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        prev_pts = np.array([p.pt for p in self.prev_kpts], dtype=np.float32).reshape((-1, 1, 2))
        pts, st, err = cv.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, prev_pts, None)
        kpts = []
        for (p, s), kp in zip(zip(pts, st), self.prev_kpts):
            if s == 1:
                new_kpt = kp
                new_kpt.pt = (p[0, 0], p[0, 1])
                kpts.append(new_kpt)
        descriptor = cv.xfeatures2d.SURF_create(400)
        new_pts, desc = descriptor.compute(frame_gray, pts)
        # pair points
        condition = np.array(st == 1)
        good_new = pts[condition]
        good_old = self.prev_pts[condition]
        # homography
        h, m = cv.findHomography(good_old, good_new, cv.RANSAC, 0)
        box = mu.transform_with_homography(h, self.prev_box)

        res = mu.draw_points(good_new, frame)
        res, box_center = mu.draw_box_homogeneous(box, res, True)
        cv.imshow('frame', res)

        # update
        self.prev_gray = frame_gray.copy()
        self.prev_pts = good_new.reshape(-1, 1, 2)
        self.prev_box = box

        if cv.waitKey(1) & 0xFF == ord('q'):
            return False
        else:
            return True

    def run_loops_webcam(self):
        cam = TrackingFromVideo(self.__startup, self.__loop)
        cam.run_loops()

    def run_loops_file(self, file_name, tracking_box):
        self.rhcw = tracking_box
        cam = TrackingFromVideo(self.__startup_file, self.__loop, file_name)
        cam.run_loops()

    def __startup_file(self, frame):
        # c. features
        self.prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vh, vw = self.prev_gray.shape
        r, h, c, w = self.rhcw
        mask, self.prev_box = mu.rhcw_to_homogeneous_box(r, h, c, w, vh, vw)
        _, p00, _ = mu.extract_features(self.prev_gray, resize=False, mask=mask)
        self.prev_pts = np.array([p.pt for p in p00], dtype=np.float32).reshape((-1, 1, 2))
        res, _ = mu.draw_box_homogeneous(self.prev_box, frame, True)
        cv.imshow('frame', res)
        # if cv.waitKey(1) & 0xFF == ord('c'):
        return False
        # else:
        #     return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv.destroyAllWindows()


with TrackingObjectFlow() as t:
    t.run_loops_webcam()
    # t.run_loops_file("images/box.mp4", (100, 310, 640, 640))
