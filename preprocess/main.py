import numpy as np

import MSRII
from c3d import C3D_conv_features


def cutting():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        dataset.write_video(dataset.take())
    print('Videos cutting done!')


def main():
    print('Loading model')
    model = C3D_conv_features(summary=True)
    print('Compiling model')
    model.compile(optimizer='sgd', loss='mse')
    print('Compiling done!')

    print('Starting extracting features')

    print('Loading mean')
    mean_total = np.load('../data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)
    print('Model done')


if __name__ == '__main__':
    main()
