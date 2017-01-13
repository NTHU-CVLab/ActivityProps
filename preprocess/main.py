import os

import h5py

import MSRII
from c3d import C3DFeatureNet


def cutting():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        dataset.write_video(dataset.take())
    print('Videos cutting done!')


def main():

    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet()

    c3dnet.start(dataset)

    # feature_file = './MSRII-c3d-features.h5'
    # mode = 'r+' if os.path.exists(feature_file) else 'w'
    # with h5py.File(feature_file, mode) as h5:
    #     print(h5.keys())
    # for name, x in get_video():
    #     x = x - mean
    #     y = model.predict(x, batch_size=32)
    #     with h5py.File(feature_file, 'r+') as f:
    #         f.create_dataset(name, data=y, dtype='float32')
    #     print('Finish extracting {}'.format(name))


if __name__ == '__main__':
    main()
