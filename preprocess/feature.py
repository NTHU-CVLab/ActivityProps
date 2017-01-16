import os
import re

import h5py
import numpy as np


class FeatureFile:

    def __init__(self, feature_file, write=False):
        self.feature_file = feature_file
        self.h5 = self.open(feature_file, write)
        self.features_keys = None
        self.labels_keys = None
        self.perm = None

    def open(self, filepath, write):
        mode = 'r+' if os.path.exists(filepath) and not write else 'w'
        return h5py.File(filepath, mode)

    def _load(self, features_keys, labels_keys):
        f = self.h5
        features = np.vstack([f.get(k) for k in features_keys])
        labels = np.concatenate([f.get(k) for k in labels_keys])
        return features, labels

    def load(self, random=False, split=0.0, **kwargs):
        f = self.h5
        features_keys = natural_sort([k for k in f.keys() if k.startswith('features')])
        labels_keys = natural_sort([k for k in f.keys() if k.startswith('labels')])
        assert len(features_keys) == len(labels_keys)
        self.features_keys = features_keys
        self.labels_keys = labels_keys

        if random and kwargs.get('video_wise'):
            _features_keys = np.array(features_keys)
            _labels_keys = np.array(features_keys)
            n = len(features_keys)
            self.perm = np.random.permutation(n)
            features_keys = _features_keys[self.perm[int(n * split):]]
            labels_keys = _labels_keys[self.perm[int(n * split):]]

        return self._load(features_keys, labels_keys)

    def save(self, features, labels, suffix):
        features_key = 'features_%s' % suffix
        labels_key = 'labels_%s' % suffix
        self.h5.create_dataset(features_key, data=features, dtype='float32')
        self.h5.create_dataset(labels_key, data=labels, dtype='int8')


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key = alphanum_key)
