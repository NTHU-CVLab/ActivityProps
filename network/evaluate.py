from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


class NetEvaluator:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.train_x, self.test_x, self.train_y, self.test_y = self.split_data()

    def split_data(self, test_size=0.2):
        return train_test_split(self.X, self.Y, test_size=0.2)

    def cross_validation(self, model_creator):
        kfold = StratifiedKFold(n_splits=4, shuffle=True)

        estimator = KerasClassifier(build_fn=model_creator, nb_epoch=10, batch_size=32, verbose=0)
        pipeline = Pipeline([
            ('standardize', StandardScaler()),
            ('mlp', estimator)
        ])

        results = cross_val_score(estimator, self.X, self.Y, cv=kfold)
        print('Accuracy: %.2f%% (+-%.2f%%)' % (results.mean() * 100, results.std() * 100))

        results = cross_val_score(pipeline, self.X, self.Y, cv=kfold)
        print('Accuracy: %.2f%% (+-%.2f%%) [Feature normalized]' % (results.mean() * 100, results.std() * 100))

    def baseline_svm(self):
        clf = svm.SVC(kernel='rbf', C=1).fit(self.train_x, self.train_y)
        accuracy = clf.score(self.test_x, self.test_y)
        print('SVM accuracy %.2f%%' % (accuracy * 100))

    def baseline_randomforest(self):
        clf = RandomForestClassifier(n_estimators=100).fit(self.train_x, self.train_y)
        accuracy = clf.score(self.test_x, self.test_y)
        print('RF accuracy %.2f%%' % (accuracy * 100))
