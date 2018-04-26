import cv2 as cv


class WebCam:
    def __init__(self, loop_fnc, flip=True) -> None:
        super().__init__()
        self.cap = cv.VideoCapture(0)
        self.loop_fnc = loop_fnc
        self.flip = flip

    def start_capture(self):
        ret = True
        while ret:
            _, frame = self.cap.read()
            if self.flip:
                frame = cv.flip(frame, 1)
            ret = self.loop_fnc(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
