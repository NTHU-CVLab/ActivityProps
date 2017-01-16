import argparse
import pickle
import re

from preprocess import MSRII
from preprocess.c3d import C3DFeatureNet
from preprocess.video import Video
from evaluate.utils import non_maximum_suppression


def extract_feature():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet(feature_file='data/features/MSRII-c3d-features.h5')
    c3dnet.load()
    c3dnet.start(dataset)

def test_proposal():
    from network.model import FC4Net

    with open('data/outputs/training_info.pkl', 'rb') as f:
        results = pickle.load(f)

    test_videos_keys = results['exclude_features_keys']
    find_vid = re.compile('features_(\d+).avi')
    video_ids = [int(find_vid.search(k).group(1)) for k in test_videos_keys]

    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet(feature_file=None)
    c3dnet.load()
    fc4net = FC4Net()
    fc4net.load_weights('data/weights/FC4Net_weights.h5')

    results = {}
    for vid in video_ids:
        video_meta = dataset.video_metas[vid]
        video = Video('/data-disk/MSRII/videos/' + video_meta.name)
        y = c3dnet.extract_feature(video)
        pred = fc4net.predict(y, batch_size=32)

        predict = non_maximum_suppression(pred, tol_frames=16)

        seg_metas = dataset.video_metas[vid].seg_metas
        target = []
        for seg_meta in seg_metas:
            s = seg_meta['start']
            t = seg_meta['duration']
            target.append((s, s + t))

        results[video_meta.name] = {
            'ground_truth': target,
            'predict': predict
        }

    with open('data/outputs/predicted_proposal.pkl', 'wb') as f:
        pickle.dump(results, f, protocol=2)  # For Python2

    # calc tIOU


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--train', action='store_true')
    p.add_argument('-e', '--extract', action='store_true',
        help='Whether in extracting feature phase.')
    p.add_argument('--eva', action='store_true')

    # For training
    p.add_argument('--videowise', action='store_true',
        help='Whether training with videowise feature splitting.')
    p.add_argument('--save', action='store_true',
        help='Whether saving training weights.')

    args = p.parse_args()

    if args.extract:
        extract_feature()
    elif args.train:
        from network.trainer import Trainer
        trainer = Trainer('data/features/MSRII-c3d-features.h5')
        trainer.run(args)
        trainer.summary()
    elif args.eva:
        test_proposal()
