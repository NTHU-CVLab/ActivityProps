import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from network.model import FC4Net


def load_data(feature_file):
    with h5py.File(feature_file, 'r') as h5:
        X = np.array(h5['features'])
        Y = np.array(h5['labels']).reshape(X.shape[0])
        Y[Y > 0] = 1
        return train_test_split(X, Y, test_size=0.2)


def train():
    feature_file = 'preprocess/MSRII-c3d-features.h5'
    X_train, X_test, y_train, y_test = load_data(feature_file)

    fc4net = FC4Net(train=True)

    fc4net.fit(
        X_train, y_train,
        nb_epoch=10, batch_size=32,
        validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = fc4net.evaluate(X_test, y_test, batch_size=32, verbose=0)
    print('Test accuracy: %.2f%%' % (accuracy * 100))

if __name__ == '__main__':
    train()
