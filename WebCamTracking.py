from WebCam import WebCam


class WebCamTracking:
    def __init__(self, startup_fnc, loop_fnc) -> None:
        super().__init__()
        self.startup_fnc = startup_fnc
        self.loop_fnc = loop_fnc

    def start_capture(self):
        with WebCam(self.startup_fnc) as cam:
            cam.run_loops()
            cam.set_loop_fnc(self.loop_fnc)
            cam.run_loops()
