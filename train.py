import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from network.model import FC1Net, FC4Net, MLPModel
from network.evaluate import NetEvaluator
from preprocess.feature import FeatureFile


def load_data(feature_file):
    X, Y = FeatureFile(feature_file).load()
    Y[Y > 0] = 1
    return X, Y


def train_test_split_videowise(feature_file):
    f = FeatureFile(feature_file)
    data = f.load(random=True, video_wise=True, split=0.1)
    X, Y = data['train']
    X_, Y_ = data['test']
    Y[Y > 0] = 1
    Y_[Y_ > 0] = 1
    print('Excluded videos: ', f.excluded)
    print('Train/Test ({}/{}) features'.format(len(Y), len(Y_)))
    return X, X_, Y, Y_


def train_fc1net(train_x, test_x, train_y, test_y, save_weights):
    fc1net = FC1Net(train=True)
    fc1net.fit(
        train_x, train_y,
        nb_epoch=500, batch_size=32,
        validation_data=(test_x, test_y), verbose=0)

    loss, accuracy = fc1net.evaluate(test_x, test_y, batch_size=32, verbose=0)
    print('=== FC1Net ===')
    print('Test accuracy: %.2f%%' % (accuracy * 100))

    if save_weights:
        fc1net.save_weights('data/weights/FC1Net_weights.h5')


def train_fc4net(train_x, test_x, train_y, test_y, save_weights):
    fc4net = FC4Net(train=True)
    fc4net.fit(
        train_x, train_y,
        nb_epoch=500, batch_size=32,
        validation_data=(test_x, test_y), verbose=0)

    loss, accuracy = fc4net.evaluate(test_x, test_y, batch_size=32, verbose=0)
    print('=== FC4Net ===')
    print('Test accuracy: %.2f%%' % (accuracy * 100))

    if save_weights:
        fc4net.save_weights('data/weights/FC4Net_weights.h5')


def train_mlp_model(train_x, test_x, train_y, test_y, save_weights):
    batch_size = 16
    nb_epoch = 500

    model = MLPModel(train=True)
    best_accuracy = 0.0
    best_epoch = 0
    np.random.seed(1993)
    for ep in range(1, nb_epoch):
        H = model.fit(
            train_x, train_y,
            batch_size=batch_size, nb_epoch=1,
            validation_split=0.2, verbose=0)
        accuracy = H.history['val_acc'][0]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = ep
            best_W = model.get_weights()

    model.reset_states()
    model.set_weights(best_W)
    print('=== MLPModel ===')
    print('best_accuracy: %f generated at epoch %d' % (best_accuracy, best_epoch))

    loss, accuracy = model.evaluate(test_x, test_y, batch_size=32, verbose=0)
    print('Test accuracy: %.2f%%' % (accuracy * 100))

    if save_weights:
        model.save_weights('data/weights/MLPModel_weights.h5')


def evaluator(train_x, test_x, train_y, test_y, X, Y):
    evaluator = NetEvaluator(X, Y)
    evaluator.X, evaluator.Y = load_data()
    evaluator.train_x, evaluator.test_x, evaluator.train_y, evaluator.test_y = train_x, test_x, train_y, test_y

    print('=== evaluator & cross-validate ===')
    evaluator.baseline_svm()
    evaluator.baseline_randomforest()
    print('-For FC1Net-')
    evaluator.cross_validation(FC1Net.build_model)
    print('-For FC4Net-')
    evaluator.cross_validation(FC4Net.build_model)
    print('-For MLPModel-')
    evaluator.cross_validation(MLPModel.build_model)


def trainer(feature_file, args):
    print('** Train under {} **'.format(feature_file))

    X, Y = load_data(feature_file)
    if args.videowise:
        train_x, test_x, train_y, test_y = train_test_split_videowise(feature_file)
    else:
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)

    train_fc1net(train_x, test_x, train_y, test_y, args.save)
    train_fc4net(train_x, test_x, train_y, test_y, args.save)
    train_mlp_model(train_x, test_x, train_y, test_y, args.save)
    evaluator(train_x, test_x, train_y, test_y, X, Y)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--videowise', action='store_true',
        help='Whether training with videowise feature splitting.')
    p.add_argument('--save', action='store_true',
        help='Whether saving training weights.')
    args = p.parse_args()

    feature_file = 'data/features/MSRII-c3d-features.h5'
    trainer(feature_file, args)
