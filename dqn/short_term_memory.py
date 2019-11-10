import numpy as np


class ShortTermMemory:

    def __init__(self, config):
        self.frames = np.zeros(
            [1, config.history_length, config.in_height, config.in_width],
            dtype=np.float32)

    def add(self, frame):
        self.frames[:, :-1, :, :] = self.frames[:, 1:, :, :]
        self.frames[:, -1, :, :] = frame
