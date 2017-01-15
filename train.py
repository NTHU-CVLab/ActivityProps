import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from network.model import FC4Net, MLPModel
from network.evaluate import NetEvaluator


def load_data(feature_file):
    with h5py.File(feature_file, 'r') as h5:
        X = np.array(h5['features'])
        Y = np.array(h5['labels']).reshape(X.shape[0])
        Y[Y > 0] = 1
        return X, Y


def train(feature_file):
    X, Y = load_data(feature_file)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    fc4net = FC4Net(train=True)
    fc4net.fit(
        X_train, y_train,
        nb_epoch=100, batch_size=32,
        validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = fc4net.evaluate(X_test, y_test, batch_size=32, verbose=0)
    print('Test accuracy: %.2f%%' % (accuracy * 100))


def train_mlp_model(feature_file):
    X, Y = load_data(feature_file)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
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
        loss, accuracy = H.history['val_loss'], H.history['val_acc']

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = ep
            best_W = model.get_weights()

    model.reset_states()
    model.set_weights(best_W)
    print('best_accuracy: %f generated at epoch %d' % (best_accuracy, best_epoch))

    loss, accuracy = model.evaluate(X_test, y_test, batch_size=16)
    print('Test accuracy: %.2f%%' % (accuracy * 100))


def evaluator(feature_file):
    X, Y = load_data(feature_file)
    evaluator = NetEvaluator(X, Y)
    evaluator.baseline_svm()
    evaluator.cross_validation(FC4Net.build_model)


if __name__ == '__main__':
    feature_file = 'data/features/MSRII-c3d-features.h5'
    train(feature_file)
    evaluator(feature_file)
