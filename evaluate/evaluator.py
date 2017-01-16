import re
import time
import pickle

import numpy as np

from preprocess.c3d import C3DFeatureNet
from preprocess.video import Video


class ProposalEvaluator:

    TRAIN_INFO = 'data/outputs/training_info.pkl'
    PREDICTED_PROPOSAL = 'data/outputs/predicted_proposal.pkl'

    def __init__(self, dataset):
        self.dataset = dataset
        self.train_info = self.load_train_info()
        self.c3dnet = C3DFeatureNet(feature_file=None)

    def eval(self, model):
        self.c3dnet.load()

        results = {}
        for vid in self.load_test_videoids():
            video_meta = self.dataset.video_metas[vid]
            video = Video('/data-disk/MSRII/videos/' + video_meta.name)

            _s = time.time()
            y = self.c3dnet.extract_feature(video)
            pred = model.predict(y, batch_size=32)
            predict = non_maximum_suppression(pred, tol_frames=16)
            print('Finish proposal for {} ({} frames) in {} sec'.format(
                video_meta.name, len(video), time.time() - _s))

            seg_metas = self.dataset.video_metas[vid].seg_metas
            target = []
            for seg_meta in seg_metas:
                s = seg_meta['start']
                t = seg_meta['duration']
                target.append((s, s + t))

            results[video_meta.name] = {
                'ground_truth': target,
                'predict': predict
            }

        self.save_proposals(results)

    def load_test_videoids(self):
        test_videos_keys = self.train_info['exclude_features_keys']
        find_vid = re.compile('features_(\d+).avi')
        video_ids = [int(find_vid.search(k).group(1)) for k in test_videos_keys]
        return video_ids

    def load_train_info(self):
        with open(self.TRAIN_INFO, 'rb') as f:
            return pickle.load(f)

    def save_proposals(self, results):
        with open(self.PREDICTED_PROPOSAL, 'wb') as f:
            pickle.dump(results, f, protocol=2)  # For Python2


def non_maximum_suppression(pred, tol_frames=16, prob_ths=.5):
    CLIP_FRAMES = 16
    p = pred.reshape(len(pred))
    candidate = np.where(p > prob_ths)

    start = np.multiply(candidate, CLIP_FRAMES).tolist()[0]
    end = (np.multiply(candidate, CLIP_FRAMES) + CLIP_FRAMES).tolist()[0]

    predict = []
    skip = [0 for _ in range(len(start))]
    for i, p in enumerate(zip(start, end)):
        s, e = p
        if skip[i]:
            continue
        next_idx = i + 1 if i + 1 < len(start) else -1
        for j, next_s in enumerate(start[next_idx:]):
            if e + tol_frames >= next_s:
                e = end[next_idx + j]
                skip[next_idx + j] = 1
        predict.append((s, e))

    return predict
