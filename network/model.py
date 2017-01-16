from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


WEIGHTS_FOLDER_FORMAT = 'data/weights/{}_weights.h5'


class BaseNet:

    NAME = 'NoneNet'

    def __init__(self, train=False, **kwargs):
        self.model = self.build_model(train)
        self.history = None
        self.epoch = 10 or kwargs.get('epoch')
        self.batch_size = 32 or kwargs.get('batch_size')
        self.verbose = 0 or kwargs.get('verbose')

    def fit(self, *args, **kwargs):
        self.history = self.model.fit(*args, **kwargs)
        return self.history

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def get_weights(self, *args, **kwargs):
        return self.model.get_weights(*args, **kwargs)

    def set_weights(self, *args, **kwargs):
        return self.model.set_weights(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.model.load_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        return self.model.save_weights(*args, **kwargs)

    def save_model(self, filename):
        model_name = filename or WEIGHTS_FOLDER_FORMAT.format(self.NAME)
        return self.model.save_weights(model_name)

    def reset_states(self):
        return self.model.reset_states()

    @staticmethod
    def build_model(self):
        raise Exception('Not implemented')

    def run(self, train, validate, save=False):
        train_x, train_y = train
        validate_x, validate_y = validate
        self.model.fit(
            train_x, train_y,
            nb_epoch=self.epoch, batch_size=self.batch_size,
            validation_data=(validate_x, validate_y),
            verbose=self.verbose)
        _, accuracy = self.model.evaluate(validate_x, validate_y, batch_size=32, verbose=0)
        if save:
            self.save_model()
        print('=== %s ===\nTest accuracy: %.2f%%' % (self.NAME, accuracy * 100))


class FC1Net(BaseNet):

    NAME = 'FC1Net'

    @staticmethod
    def build_model(train=True):
        model = Sequential()
        model.add(Dense(1, input_dim=4096, activation='sigmoid'))

        if train:
            model.compile(
                optimizer='rmsprop',
                loss='binary_crossentropy',
                metrics=['accuracy'])

        return model


class FC4Net(BaseNet):

    NAME = 'FC4Net'

    @staticmethod
    def build_model(train=True):
        model = Sequential()
        model.add(Dense(256, input_dim=4096, init='uniform', activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        if train:
            model.compile(
                loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

        return model


class MLPModel(BaseNet):

    NAME = 'MLPModel'

    @staticmethod
    def build_model(train=True):
        model = Sequential()
        model.add(Dense(256, input_dim=4096, init='uniform', activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(256, init='uniform', activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(256, init='uniform', activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(1, init='uniform', activation='sigmoid'))

        if train:
            sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(
                loss='binary_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])

        return model

    def run(self, train, validate, save=False):
        train_x, train_y = train
        validate_x, validate_y = validate
        model = self.model

        best_accuracy, best_epoch = 0.0, 0
        for ep in range(self.epoch):
            H = model.fit(
                train_x, train_y,
                batch_size=self.batch_size, nb_epoch=1,
                validation_split=0.2, verbose=self.verbose)
            accuracy = H.history['val_acc'][0]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_epoch = ep
                best_W = model.get_weights()

        model.reset_states()
        model.set_weights(best_W)
        self.model = model

        _, accuracy = self.model.evaluate(validate_x, validate_y, batch_size=32, verbose=0)

        print('=== %s ===\nTest accuracy: %.2f%%' % (self.NAME, accuracy * 100))
        print('best_accuracy: %f generated at epoch %d' % (best_accuracy, best_epoch))

        if save:
            self.save_model()
