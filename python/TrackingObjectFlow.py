import cv2 as cv
import numpy as np

from python import matech_utilities as mu
from python.TrackingFromVideo import TrackingFromVideo


class TrackingObjectFlow:
    sigma = 51

    def __init__(self, record=True) -> None:
        super().__init__()
        self.record = record

    def __loop(self, frame):
        # Optic flow
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        prev_pts = np.array([p.pt for p in self.prev_kpts], dtype=np.float32).reshape((-1, 1, 2))
        opts, st, err = cv.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, prev_pts, None)
        condition = st == 1
        pts = opts[condition]
        prev_pts = prev_pts[condition]

        kpts = mu.points_to_keypoints(pts, self.prev_kpts)
        descriptor = cv.xfeatures2d.SURF_create(400)
        kpts, feat = descriptor.compute(frame_gray, kpts)
        # pair points
        matches = mu.match_features(self.org_feat, feat)
        box = None
        if len(matches) > 20:
            h, m = mu.find_homography(self.org_pts, pts, matches)
            if h is not None:
                box = mu.transform_with_homography(h, self.org_box)

        if box is None:
            h, m = cv.findHomography(prev_pts, pts, cv.RANSAC, 3.0)
            if h is not None:
                box = mu.transform_with_homography(h, self.prev_box)

        if box is not None:
            res = self.print_points(frame, opts.reshape((-1, 2)), condition)
            res = self.print_points(res, pts, m)
            # box = self.recenter_box(box, pts, m)
            res, box_center = mu.draw_box_homogeneous(box, res, True)
            self.prev_box = box
        else:
            res = frame

        if self.record:
            self.out.write(res)
        cv.imshow('frame', res)

        # update
        self.prev_gray = frame_gray.copy()
        self.prev_kpts = kpts

        if cv.waitKey(1) & 0xFF == ord('q'):
            return False
        else:
            return True

    def recenter_box(self, box, pts, m):
        a = []
        for p, g in zip(pts, m):
            if g == 1:
                a.append(p)
        cx = 0
        cy = 0
        for p in a:
            cx += p[0]
            cy += p[1]
        cx /= len(a)
        cy /= len(a)
        print(cx, cy)
        r = np.mean(box, axis=0)
        ocx = r[0]
        ocy = r[1]
        box[:, :2] -= int(ocx - cx)
        box[:, :2] -= int(ocy - cy)
        return box

    def print_points(self, res, pts, m):
        gp = []
        bp = []
        for p, g in zip(pts, m):
            if g == 1:
                gp.append(p)
            else:
                bp.append(p)
        res = mu.draw_points(gp, res)
        res = mu.draw_points(bp, res, (125, 125, 125))
        return res

    def run_loops_webcam(self):
        cam = TrackingFromVideo(self.__startup_webcam, self.__loop)
        cam.run_loops()

    def run_loops_file(self, file_name, tracking_box):
        self.rhcw = tracking_box
        cam = TrackingFromVideo(self.__startup_file, self.__loop, file_name)
        if self.record:
            self.out = None
        cam.run_loops()

    def __startup_webcam(self, frame):
        box, mask = mu.shape_to_homogeneous_box(frame.shape)
        res, _ = mu.draw_box_homogeneous(box, frame, True)

        cv.imshow('frame', res)

        if cv.waitKey(1) & 0xFF == ord('c'):
            self.set_up_frame_data(frame, mask, box)
            return False
        else:
            return True

    def set_up_frame_data(self, frame, mask, box):
        self.prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, org_kpts, self.org_feat = mu.extract_features(self.prev_gray, resize=False, mask=mask)
        self.org_pts = [p.pt for p in org_kpts]
        self.prev_kpts = org_kpts
        self.prev_box = box
        self.org_box = box
        if self.record:
            fourcc = cv.VideoWriter_fourcc(*'DIVX')
            h, w, _ = frame.shape
            self.out = cv.VideoWriter('output.mkv', fourcc, 25.0, (w, h), isColor=True)

    def __startup_file(self, frame):
        # c. features
        self.prev_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        vh, vw = self.prev_gray.shape
        r, h, c, w = self.rhcw
        mask, box = mu.rhcw_to_homogeneous_box(r, h, c, w, vh, vw)
        self.set_up_frame_data(frame, mask, box)
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cv.destroyAllWindows()
        self.out.release()


with TrackingObjectFlow() as t:
    # t.run_loops_webcam()
    t.run_loops_file("images/ger2.mp4", (230, 270, 920, 180))
