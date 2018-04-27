import cv2 as cv

import matech_utilities as mu
from TrackingFromVideo import TrackingFromVideo


class TrackingFeature:
    MIN_MATCH_COUNT = 10

    def __init__(self) -> None:
        super().__init__()

    def __startup(self, frame):
        self.org_box, mask = mu.shape_to_homogeneous_box(frame.shape)

        res, _ = mu.draw_box_homogeneous(self.org_box, frame, True)

        # Display the resulting frame
        cv.imshow('frame', res)
        if cv.waitKey(1) & 0xFF == ord('c'):
            # c. features
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, self.org_pts, self.org_feat = mu.extract_features(gray, resize=False, mask=mask)
            return False
        else:
            return True

    def __loop(self, frame):
        _, pts, feat = mu.extract_features(frame, False)
        matches = mu.match_features(self.org_feat, feat)
        # print(len(matches))
        if len(matches) > self.MIN_MATCH_COUNT:
            h = mu.find_homography([p.pt for p in self.org_pts], [p.pt for p in pts], matches)
            cs = mu.transform_with_homography(h, self.org_box)
            res, center = mu.draw_box_homogeneous(cs, frame, True)
            cv.imshow('frame', res)
            if cv.waitKey(1) & 0xFF == ord('q'):
                return False
        return True

    def run_loops(self):
        cam = TrackingFromVideo(self.__startup, self.__loop)
        cam.run_loops()
        cv.destroyAllWindows()


TrackingFeature().run_loops()
