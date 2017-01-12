import numpy as np

import MSRII
# from c3d import C3D_conv_features


def cutting():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        dataset.write_video(dataset.take())
    print('Videos cutting done!')


def test():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    for _ in range(MSRII.Dataset.NUM_VIDEOS):
        video = dataset.read_cutting_video()
        video = video.transpose(1, 0, 2, 3)
        num_frames = video.shape[0]
        print(video.shape)
        break


def main():
    model = C3D_conv_features(summary=True)
    model.compile(optimizer='sgd', loss='mse')
    print('Network ready!')
    mean_total = np.load('../data/models/c3d-sports1M_mean.npy')
    mean = np.mean(mean_total, axis=(0, 2, 3, 4), keepdims=True)
    print('Model done')
    import pdb
    pdb.set_trace()

    print('Starting extracting features')


if __name__ == '__main__':
    test()
    # main()
