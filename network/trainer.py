import pickle

from sklearn.model_selection import train_test_split

from preprocess.feature import FeatureFile
from network.model import FC1Net, FC4Net, MLPModel
from network.evaluate import NetEvaluator


class Trainer:

    def __init__(self, feature_file):
        self.exclude_features_keys = None
        self.feature_filename = feature_file
        self.X, self.Y = self.load_full_data(feature_file)
        self.feature_file = FeatureFile(feature_file)

    def run(self, args):
        print('** Train under {} **'.format(self.feature_filename))

        if args.videowise:
            train_x, test_x, train_y, test_y = self.train_test_split_videowise()
        else:
            train_x, test_x, train_y, test_y = train_test_split(self.X, self.Y, test_size=0.2)

        FC1Net(train=True).run((train_x, train_y), (test_x, test_y), args.save)
        FC4Net(train=True).run((train_x, train_y), (test_x, test_y), args.save)
        MLPModel(train=True).run((train_x, train_y), (test_x, test_y), args.save)

        self.evaluator((train_x, train_y), (test_x, test_y), self.X, self.Y)

    def evaluator(self, train, test, X, Y):
        train_x, train_y = train
        test_x, test_y = test

        evaluator = NetEvaluator(X, Y)
        evaluator.X, evaluator.Y = self.X, self.Y
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

    def load_full_data(self, feature_file):
        X, Y = FeatureFile(feature_file).load()
        Y[Y > 0] = 1
        return X, Y

    def train_test_split_videowise(self):
        f = self.feature_file
        data = f.load(random=True, video_wise=True, split=0.1)
        X, Y = data['train']
        X_, Y_ = data['test']
        Y[Y > 0] = 1
        Y_[Y_ > 0] = 1
        print('Excluded videos: ', f.excluded)
        print('Train/Test ({}/{}) features'.format(len(Y), len(Y_)))
        self.exclude_features_keys = f.excluded
        return X, X_, Y, Y_

    def summary(self):
        results = {
            'exclude_features_keys': self.exclude_features_keys
        }
        with open('data/outputs/training_info.pkl', 'wb') as f:
            pickle.dump(results, f, protocol=2)  # For Python2
