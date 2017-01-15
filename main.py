import numpy as np

from preprocess import MSRII
from preprocess.c3d import C3DFeatureNet
from preprocess.video import Video
from network.model import FC4Net


def main():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet(feature_file='data/features/MSRII-c3d-features.h5')

    c3dnet.load()
    c3dnet.start(dataset)


def test(num_video=5):
    dataset = MSRII.Dataset('/data-disk/MSRII/')

    c3dnet = C3DFeatureNet(feature_file=None)
    c3dnet.load()

    fc4net = FC4Net()
    fc4net.load_weights('data/weights/FC4Net_weights.h5')

    for i in range(num_video):
        video_meta = dataset.video_metas[i]
        video = Video('/data-disk/MSRII/videos/' + video_meta.name)
        y = c3dnet.extract_feature(video)
        pred = fc4net.predict(y, batch_size=32)

        predict = nms(pred, tol=16)

        seg_metas = dataset.video_metas[0].seg_metas
        target = []
        for seg_meta in seg_metas:
            s = seg_meta['start']
            t = seg_meta['duration']
            target.append((s, s + t))

        print('Target', target)
        print('Predict', predict)


def nms(pred, tol=16, prob_ths=.5):
    p = pred.reshape(len(pred))
    candidate = np.where(p > prob_ths)

    start = np.multiply(candidate, 16).tolist()[0]
    end = (np.multiply(candidate, 16) + 16).tolist()[0]

    predict = []
    skip = [0 for _ in range(len(start))]
    for i, p in enumerate(zip(start, end)):
        s, e = p
        if skip[i]:
            continue
        next_idx = i + 1 if i + 1 < len(start) else -1
        for j, next_s in enumerate(start[next_idx:]):
            if e + tol >= next_s:
                e = end[next_idx + j]
                skip[next_idx + j] = 1
        predict.append((s, e))

    return predict


if __name__ == '__main__':
    test(num_video=5)
