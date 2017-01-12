import itertools
from collections import namedtuple

from data import Video

Meta = namedtuple('VideoMeta', ['name', 'metas'])


class Dataset:

    LABEL_FILE = 'MSRHA_Dataset_GT_160x120.txt'
    LABEL_FILE_HEAD = 6
    VIDEO_FOLDER = 'videos/'

    def __init__(self, root):
        self.root = root
        self.video_folder = root + self.VIDEO_FOLDER
        self.metas = self.load_label()

        self._seek = 0

    @property
    def seek(self):
        p = self._seek
        self._seek += 1
        return p

    def load_label(self):
        with open(self.root + self.LABEL_FILE) as f:
            raw = [
                self._build_meta(line.split())
                for line in f.readlines()[self.LABEL_FILE_HEAD:]
            ]
        return [
            Meta(video, metas)
            for video, metas in itertools.groupby(raw, lambda x: x['name'])
        ]

    def _build_meta(self, tokens):
        return {
            'name': tokens[0][1:-1],
            'left': int(tokens[1]),
            'width': int(tokens[2]),
            'top': int(tokens[3]),
            'height': int(tokens[4]),
            'start': int(tokens[5]),
            'duration': int(tokens[6]),
            'class': int(tokens[7]),
        }

    def take(self):
        return self._read_video(self.metas[self.seek])

    def _read_video(self, meta):
        filepath = '{}{}'.format(self.video_folder, meta.name)
        with Video(filepath) as video:
            return meta.name, video.read()
