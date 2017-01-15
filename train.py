import h5py
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from network.model import FC4Net


def load_data(feature_file):
    with h5py.File(feature_file, 'r') as h5:
        X = np.array(h5['features'])
        Y = np.array(h5['labels']).reshape(X.shape[0])
        Y[Y > 0] = 1
        return train_test_split(X, Y, test_size=0.2)


def train(feature_file):
    X_train, X_test, y_train, y_test = load_data(feature_file)

    fc4net = FC4Net(train=True)

    fc4net.fit(
        X_train, y_train,
        nb_epoch=10, batch_size=32,
        validation_data=(X_test, y_test), verbose=0)

    loss, accuracy = fc4net.evaluate(X_test, y_test, batch_size=32, verbose=0)
    print('Test accuracy: %.2f%%' % (accuracy * 100))


def cross_validation(feature_file, model_creator):
    X_train, X_test, y_train, y_test = load_data(feature_file)
    kfold = StratifiedKFold(n_splits=4, shuffle=True)

    estimator = KerasClassifier(
        build_fn=model_creator, nb_epoch=10, batch_size=32, verbose=0)
    pipeline = Pipeline([
        ('standardize', StandardScaler()),
        ('mlp', KerasClassifier(
            build_fn=model_creator, nb_epoch=10, batch_size=32, verbose=0))
    ])

    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print('Accuracy: %.2f%% (+-%.2f%%)' % (
        results.mean() * 100, results.std() * 100))

    results = cross_val_score(pipeline, X_train, y_train, cv=kfold)
    print('Accuracy: %.2f%% (+-%.2f%%) [Feature normalized]' % (
        results.mean() * 100, results.std() * 100))


def baselines(feature_file):
    X_train, X_test, y_train, y_test = load_data(feature_file)
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier

    clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('SVM accuracy %.2f%%' % (accuracy * 100))

    clf = RandomForestClassifier(n_estimators=100).fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('RF accuracy %.2f%%' % (accuracy * 100))


if __name__ == '__main__':
    feature_file = 'preprocess/MSRII-c3d-features.h5'

    train(feature_file)
    cross_validation(feature_file, FC4Net.build_model)
    baselines(feature_file)
