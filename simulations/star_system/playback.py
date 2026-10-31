"""Forward recorded-time sample selection; this module never calls the solver."""
from bisect import bisect_right
import math


class Playback:
    def __init__(self, recording):
        self.recording = recording
        self.index = 0
        self.playing = False
        self._times = tuple(frame.time for frame in recording.frames)
        self._target_time = self._times[0]

    @property
    def current_frame(self):
        return self.recording.frames[self.index]

    def play(self):
        self.playing = self.index < len(self._times)-1

    def pause(self):
        self.playing = False

    def next_sample(self):
        self.pause()
        self.index = min(self.index+1, len(self._times)-1)
        self._target_time = self._times[self.index]
        return self.current_frame

    def advance(self, wall_seconds, years_per_second):
        if not math.isfinite(wall_seconds) or wall_seconds < 0 or not math.isfinite(years_per_second) or years_per_second <= 0:
            raise ValueError('Elapsed wall seconds must be nonnegative; playback yr/sec must be positive and finite.')
        if self.playing:
            self._target_time = min(self._times[-1], self._target_time+wall_seconds*years_per_second)
            self.index = max(self.index, bisect_right(self._times, self._target_time)-1)
            if self.index == len(self._times)-1:
                self.pause()
        return self.current_frame
