import argparse

from preprocess import MSRII, KTH
from preprocess.c3d import C3DFeatureNet
from network.trainer import Trainer
from network.model import FC4Net
from evaluate.evaluator import ProposalEvaluator


def extract_feature():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    c3dnet = C3DFeatureNet(feature_file='data/features/MSRII-c3d-features.h5')
    c3dnet.load()
    c3dnet.start(dataset)

    dataset = KTH.Dataset('/data-disk/KTH/')
    c3dnet = C3DFeatureNet(feature_file='data/features/KTH-c3d-features.h5')
    c3dnet.load()
    c3dnet.start(dataset)


def train_models(args):
    trainer = Trainer(feature_file='data/features/MSRII-c3d-features.h5')
    trainer = Trainer(feature_file='data/features/KTH-c3d-features.h5')
    trainer.run(args)
    # trainer.run(args, extra_test='data/features/MSRII-c3d-features.h5')
    trainer.summary()


def generate_proposal():
    dataset = MSRII.Dataset('/data-disk/MSRII/')
    fc4net = FC4Net()
    fc4net.load_weights('data/weights/FC4Net_weights.h5')

    evaluatoor = ProposalEvaluator(dataset)
    evaluatoor.eval(fc4net)
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
        train_models(args)
    elif args.eva:
        generate_proposal()
