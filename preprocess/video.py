import cv2
import numpy as np
from scipy.misc import imresize


class Video:

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)
        self.frames = None

    def __enter__(self):
        if not self.cap.isOpened():
            raise Exception('Cannot open video: {}'.format(self.path))
        return self

    def __len__(self):
        if not self.frames:
            self.load()
        return len(self.frames)

    def load(self):
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        self.frames = frames
        return self

    def read(self):
        return self.load().frames

    def resize(self, size=None):
        if size:
            self.frames = [imresize(f, (size[1], size[0])) for f in self.frames]
        return self.frames

    @classmethod
    def np_array(cls, frames, dim_ordering='th'):
        video = np.array(frames, dtype=np.float32)
        if dim_ordering == 'th':
            video = video.transpose(3, 0, 1, 2)
        return video

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()


def save_video(filepath, fps, w, h, data):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for frame in data:
        out.write(frame)
    out.release()
