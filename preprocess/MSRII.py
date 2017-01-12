import itertools
from collections import namedtuple

from video import Video, save_video

Meta = namedtuple('VideoMeta', ['name', 'metas'])


class Dataset:

    LABEL_FILE = 'MSRHA_Dataset_GT_160x120.txt'
    LABEL_FILE_HEAD = 6
    VIDEO_FOLDER = 'videos/'

    VIDEO_WIDTH = 320
    VIDEO_HEIGHT = 240
    VIDEO_FRAMERATE = 15

    NUM_VIDEOS = 54

    def __init__(self, root):
        self.root = root
        self.video_folder = root + self.VIDEO_FOLDER
        self.video_metas = self.load_label()

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
            Meta(video, list(metas))
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

    def read_cutting_video(self, folder='cutting/'):
        cutting_video_folder = self.root + folder
        name, _ = self.video_metas[self.seek]
        with Video(cutting_video_folder + name) as v:
            return name, v.np_read(resized_size=(112, 112))

    def take(self):
        return self._read_video(self.video_metas[self.seek])

    def _read_video(self, video):
        with Video(self.video_folder + video.name) as v:
            frames = v.read()
        needed_frames = []
        for meta in video.metas:
            s = meta['start']
            t = meta['duration']
            needed_frames += frames[s:s + t]
        return video.name, needed_frames

    def write_video(self, video, output_dir='cutting/'):
        name, frames = video
        save_video(
            self.root + output_dir + name,
            self.VIDEO_FRAMERATE, self.VIDEO_WIDTH, self.VIDEO_HEIGHT,
            frames
        )
