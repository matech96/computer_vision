from VideoReader import VideoReader


class TrackingFromVideo:
    def __init__(self, startup_fnc, loop_fnc, file_name=0) -> None:
        super().__init__()
        self.startup_fnc = startup_fnc
        self.loop_fnc = loop_fnc
        self.file_name = file_name
        self.flip = file_name == 0

    def run_loops(self):
        with VideoReader(self.startup_fnc, self.file_name, self.flip) as cam:
            cam.run_loops()
            cam.set_loop_fnc(self.loop_fnc)
            cam.run_loops()
