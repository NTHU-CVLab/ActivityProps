import os
import itertools
from collections import namedtuple

from video import Video, save_video

Meta = namedtuple('VideoMeta', ['name', 'seg_metas'])
Segments = namedtuple('VideoSegments', ['frames', 'label'])


class Dataset:

    VIDEO_FOLDER = 'videos/'
    LABEL_FILE = 'MSRHA_Dataset_GT_160x120.txt'
    LABEL_FILE_HEAD = 6

    VIDEO_WIDTH = 320
    VIDEO_HEIGHT = 240
    VIDEO_FRAMERATE = 15

    NUM_VIDEOS = 54
    OTHER_LABEL = 0

    def __init__(self, root):
        self.root = root
        self.video_folder = os.path.join(root, self.VIDEO_FOLDER)
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
            Meta(video, list(seg_metas))
            for video, seg_metas in itertools.groupby(raw, lambda x: x['name'])
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

    def get(self, resized_size=None):
        for video_meta in self.video_metas:
            frames = self.read_video(video_meta, resized_size)
            yield self.split_by_any_tag(video_meta, frames)

    def read_video(self, video_meta, resized_size):
        with Video(self.video_folder + video_meta.name) as v:
            return v.read(resized_size)

    def split_by_any_tag(self, video_meta, frames):
        seg_metas = video_meta.seg_metas
        s = [m['start'] for m in seg_metas]
        t = [m['duration'] for m in seg_metas]
        class_map = {m['start']: m['class'] for m in seg_metas}

        last = len(frames)
        start_frames = sorted(s + [a + b for a, b in zip(s, t)] + [0, last])
        stop_frames = start_frames[1:]

        return [
            Segments(frames[start:stop], class_map.get(start, self.OTHER_LABEL))
            for start, stop in zip(start_frames, stop_frames)
        ]

    def read_cutting_video(self, folder='cutting/'):
        folder = os.path.join(self.root, folder)
        name, _ = self.video_metas[self.seek]
        with Video(folder + name) as v:
            return name, v.np_read(resized_size=(112, 112))

    def take(self):
        return self._read_video_dep(self.video_metas[self.seek])

    def _read_video_dep(self, video):
        with Video(self.video_folder + video.name) as v:
            frames = v.read()
        needed_frames = []
        for meta in video.seg_metas:
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
