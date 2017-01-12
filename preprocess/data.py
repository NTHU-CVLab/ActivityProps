import cv2


class Video:

    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(self.path)

    def __enter__(self):
        if not self.cap.isOpened():
            raise Exception('Cannot open video: {}'.format(self.path))
        return self

    def read(self, duration=None):
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
        return frames

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()


def save_video(filepath, fps, w, h, data):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
    for frame in data:
        out.write(frame)
    out.release()
