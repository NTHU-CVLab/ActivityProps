import os
import itertools
from collections import namedtuple

from preprocess.video import Video, save_video

Meta = namedtuple('VideoMeta', ['name', 'seg_metas'])
Segments = namedtuple('VideoSegments', ['frames', 'label'])

class Dataset:

    VIDEO_FOLDER = 'videos/'
    LABEL_FILE = '00sequences.txt'
    LABEL_FILE_HEAD = 0

    VIDEO_WIDTH = 160
    VIDEO_HEIGHT = 120
    VIDEO_FRAMERATE = 25

    NUM_VIDEOS = 599

    OTHER_LABEL = 0
    CLASS_LABELS_MAP = {
        'boxing': 3,
        'handclapping': 1,
        'handwaving': 2,
        'jogging': 4,
        'running': 5,
        'walking': 6
    }

    def __init__(self, root):
        self.root = root
        self.video_folder = os.path.join(root, self.VIDEO_FOLDER)
        self.sequence_count = 0
        self.video_metas = self.load_label()

    def load_label(self):
        with open(self.root + self.LABEL_FILE) as f:
            raw = []
            for line in f.readlines()[self.LABEL_FILE_HEAD:]:
                for meta in self._build_meta(line.split()):
                    raw.append(meta)

        self.raw = raw
        return [
            Meta(video, list(seg_metas))
            for video, seg_metas in itertools.groupby(raw, lambda x: x['name'])
        ]

    def _build_meta(self, tokens):
        more_data_keyword = 'frames'

        if len(tokens) < 1:
            return

        result = {'name': tokens[0] + '_uncomp.avi'}
        more_data_notation = tokens[1]

        if more_data_keyword in more_data_notation:
            for left_token in tokens[2:]:
                start, end = left_token.split('-')
                result['start'] = int(start)
                result['duration'] = int(end.split(',')[0]) - int(start) + 1

                class_type = result['name'].split('_')[1]
                result['class'] = self.CLASS_LABELS_MAP[class_type]

                self.sequence_count += 1

                yield result

    def get(self, resized_size=None):
        for video_meta in self.video_metas:
            frames = self.read_video(video_meta, resized_size)
            yield self.split_by_any_tag(video_meta, frames)

    def read_video(self, video_meta, size=None):
        with Video(self.video_folder + video_meta.name) as v:
            return v.load().resize(size)

    def split_by_any_tag(self, video_meta, frames):
        seg_metas = video_meta.seg_metas
        s = [m['start'] for m in seg_metas]
        t = [m['duration'] for m in seg_metas]
        label_map = {m['start']: m['class'] for m in seg_metas}

        last = len(frames)
        start_frames = sorted(s + [a + b for a, b in zip(s, t)] + [0, last])
        stop_frames = start_frames[1:]

        return video_meta.name, [
            Segments(frames[start:stop],
                     label_map.get(start, self.OTHER_LABEL))
            for start, stop in zip(start_frames, stop_frames)
        ]

    def write_video(self, video, output_dir='cutting/'):
        name, frames = video
        save_video(self.root + output_dir + name, self.VIDEO_FRAMERATE,
                   self.VIDEO_WIDTH, self.VIDEO_HEIGHT, frames)
