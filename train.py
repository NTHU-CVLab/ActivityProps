import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from network.model import FC1Net, FC4Net, MLPModel
from network.evaluate import NetEvaluator


def load_data(feature_file):
    with h5py.File(feature_file, 'r') as h5:
        X = np.array(h5['features'])
        Y = np.array(h5['labels']).reshape(X.shape[0])
        Y[Y > 0] = 1
        return X, Y


def load_test_data():
    with h5py.File('data/features/MSRII-c3d-features.h5', 'r') as h5:
        X = np.array(h5['features'])
        Y = np.array(h5['labels']).reshape(X.shape[0])
        Y[Y > 0] = 1
        return X, Y


def _train_test_split(feature_file):
    train_x, train_y = load_data(feature_file)
    test_x, test_y = load_test_data()
    test_x = test_x[:len(test_x) - len(train_x)]
    test_y = test_y[:len(test_y) - len(train_y)]
    return train_x, test_x, train_y, test_y


def train_fc1net(feature_file):
    # X, Y = load_data(feature_file)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test, y_train, y_test = _train_test_split(feature_file)

    fc1net = FC1Net(train=True)
    fc1net.fit(
        X_train, y_train,
        nb_epoch=500, batch_size=32,
        validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = fc1net.evaluate(X_test, y_test, batch_size=32, verbose=0)
    print('=== FC1Net ===')
    print('Test accuracy: %.2f%%' % (accuracy * 100))

    fc1net.save_weights('data/weights/FC1Net_weights.h5')


def train_fc4net(feature_file):
    # X, Y = load_data(feature_file)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test, y_train, y_test = _train_test_split(feature_file)

    fc4net = FC4Net(train=True)
    fc4net.fit(
        X_train, y_train,
        nb_epoch=500, batch_size=32,
        validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = fc4net.evaluate(X_test, y_test, batch_size=32, verbose=0)
    print('=== FC4Net ===')
    print('Test accuracy: %.2f%%' % (accuracy * 100))

    fc4net.save_weights('data/weights/FC4Net_weights.h5')


def train_mlp_model(feature_file):
    # X, Y = load_data(feature_file)
    # X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_test, y_train, y_test = _train_test_split(feature_file)
    batch_size = 16
    nb_epoch = 1000

    model = MLPModel(train=True)
    best_accuracy = 0.0
    best_epoch = 0
    np.random.seed(1993)
    for ep in range(1, nb_epoch):
        H = model.fit(
            X_train, y_train,
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

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=16, verbose=0)
    print('Test accuracy: %.2f%%' % (accuracy * 100))

    model.save_weights('data/weights/MLPModel_weights.h5')


def evaluator(feature_file):
    X, Y = load_data(feature_file)
    evaluator = NetEvaluator(X, Y)
    evaluator.X, evaluator.Y = load_test_data()
    evaluator.train_x, evaluator.test_x, evaluator.train_y, evaluator.test_y = _train_test_split(feature_file)

    print('=== evaluator & cross-validate ===')
    evaluator.baseline_svm()
    evaluator.baseline_randomforest()
    print('-For FC4Net-')
    evaluator.cross_validation(FC4Net.build_model)
    print('-For MLPModel-')
    evaluator.cross_validation(MLPModel.build_model)


if __name__ == '__main__':
    feature_file = 'data/features/MSRII-c3d-features-excluded-first5.h5'
    print('** Train under {} **'.format(feature_file))

    train_fc1net(feature_file)
    train_fc4net(feature_file)
    # train_mlp_model(feature_file)
    evaluator(feature_file)
