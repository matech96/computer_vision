import cv2 as cv


class VideoReader:
    def __init__(self, loop_fnc, file_name=0, flip=True) -> None:
        super().__init__()
        self.cap = cv.VideoCapture(file_name)
        self.loop_fnc = loop_fnc
        self.flip = flip

    def set_loop_fnc(self, loop_fnc):
        self.loop_fnc = loop_fnc

    def run_loops(self):
        ret = True
        while ret:
            has_frame, frame = self.cap.read()
            if not has_frame:
                break

            if self.flip:
                frame = cv.flip(frame, 1)
            ret = self.loop_fnc(frame)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
