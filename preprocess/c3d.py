import time

import h5py
import numpy as np
from keras import backend as K
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

from video import Video


class C3DFeatureNet:

    MODEL_WEIGHTS = '../data/models/c3d-sports1M_weights.h5'
    MODEL_MEAN = '../data/models/c3d-sports1M_mean.npy'

    INPUT_FRAMES = 16

    def __init__(self, feature_file, model_weight=None, model_mean=None):
        self.model_weight = model_weight or self.MODEL_WEIGHTS
        self.model_mean = model_mean or self.MODEL_MEAN
        self.input_size = (112, 112)
        self.feature_file = feature_file
        self.model = None
        self.mean = None

    def load(self):
        self.model = self.load_network()
        self.mean = self.load_mean()
        self.model.compile(optimizer='sgd', loss='mse')
        print('C3D-Network is ready.')

    def start(self, dataset):
        model = self.model
        mean = self.mean

        def apply_model(seg):
            num_x, x = self.build_input(seg.frames)
            return [seg.label] * num_x, model.predict(x - mean, batch_size=32)

        labels, features = [], []
        for i, item in enumerate(dataset.get(self.input_size)):
            name, segs = item
            s = time.time()

            packs = [apply_model(seg) for seg in segs if len(seg.frames)]
            labels.append(np.vstack([l for pack in packs for l in pack[0]]))
            features.append(np.vstack([f for pack in packs for f in pack[1]]))

            print('Finish {} in {} sec.'.format(name, time.time() - s))

        self.save_features(np.vstack(features), np.vstack(labels))

    def extract_feature(self, video):
        model = self.model
        mean = self.mean

        def apply_model(frames):
            _, x = self.build_input(frames)
            return model.predict(x - mean, batch_size=32)

        return apply_model(video.load().resize(self.input_size))

    def build_input(self, frames):
        np_video = Video.np_array(frames).transpose(1, 0, 2, 3)

        num_frames = np_video.shape[0]
        num_clips = num_frames // self.INPUT_FRAMES

        np_video = np_video[:num_clips * self.INPUT_FRAMES, :, :, :]
        np_video = np_video.reshape(
                    (num_clips, self.INPUT_FRAMES, 3,) + self.input_size)
        np_video = np_video.transpose(0, 2, 1, 3, 4)

        return num_clips, np_video

    def save_features(self, features, labels):
        with h5py.File(self.feature_file, 'w') as h5:
            h5.create_dataset('features', data=features, dtype='float32')
            h5.create_dataset('labels', data=labels, dtype='int8')

    def load_mean(self):
        mean_total = np.load(self.model_mean)
        return np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)

    def load_network(self, summary=False):
        K.set_image_dim_ordering('th')

        model = Sequential()
        # 1st layer group
        model.add(Convolution3D(64, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv1',
                                subsample=(1, 1, 1),
                                input_shape=(3, 16, 112, 112),
                                trainable=False))
        model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                            border_mode='valid', name='pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv2',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                            border_mode='valid', name='pool2'))
        # 3rd layer group
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3a',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(Convolution3D(256, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv3b',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                            border_mode='valid', name='pool3'))
        # 4th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4a',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv4b',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                            border_mode='valid', name='pool4'))
        # 5th layer group
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5a',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(Convolution3D(512, 3, 3, 3, activation='relu',
                                border_mode='same', name='conv5b',
                                subsample=(1, 1, 1),
                                trainable=False))
        model.add(ZeroPadding3D(padding=(0, 1, 1), name='zeropadding'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                            border_mode='valid', name='pool5'))
        model.add(Flatten(name='flatten'))
        # FC layers group
        model.add(Dense(4096, activation='relu', name='fc6', trainable=False))
        model.add(Dropout(.5, name='do1'))
        model.add(Dense(4096, activation='relu', name='fc7'))
        model.add(Dropout(.5, name='do2'))
        model.add(Dense(487, activation='softmax', name='fc8'))

        # Load weights
        model.load_weights(self.model_weight)

        for _ in range(4):
            model.pop()

        if summary:
            print(model.summary())
        return model
