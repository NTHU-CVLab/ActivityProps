import cv2
import numpy as np
import scipy.misc


class Video:

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)

    def __enter__(self):
        if not self.cap.isOpened():
            raise Exception('Cannot open video: {}'.format(self.path))
        return self

    def read(self, resize=None):
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if resize:
                frame = scipy.misc.imresize(frame, (resize[1], resize[0]))
            frames.append(frame)
        return frames

    def np_read(self, resized_size, dim_ordering='th'):
        video = np.array(self.read(resized_size), dtype=np.float32)
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
