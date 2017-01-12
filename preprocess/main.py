import h5py
import numpy as np

import MSRII
from c3d import C3D_conv_features


def cutting():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        dataset.write_video(dataset.take())
    print('Videos cutting done!')


def get_video():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        name, video = dataset.read_cutting_video()

        video = video.transpose(1, 0, 2, 3)
        num_frames = video.shape[0]
        num_instances = num_frames // 16
        video = video[:num_instances*16, :, :, :]
        video = video.reshape((num_instances, 16, 3,) + (112, 112))
        video = video.transpose(0, 2, 1, 3, 4)

        yield name, video


def main():
    model = C3D_conv_features(summary=True)
    model.compile(optimizer='sgd', loss='mse')
    print('Network ready!')
    mean_total = np.load('../data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)
    print('Model done')

    print('Starting extracting features')
    for name, x in get_video():
        x = x - mean
        y = model.predict(x, batch_size=32)
        with h5py.File('./MSRII-c3d-features.h5', 'r+') as f:
            f.create_dataset(name, data=y, dtype='float32')
        print('Finish extracting {}'.format(name))


if __name__ == '__main__':
    main()
