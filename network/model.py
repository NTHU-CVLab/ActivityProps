from keras.models import Sequential
from keras.layers import Dense, Dropout


class BaseNet:

    def __init__(self, train=False):
        self.model = self.build_model(train)
        self.history = None

    def fit(self, *args, **kwargs):
        self.history = self.model.fit(*args, **kwargs)
        return self.history

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        return self.model.evaluate(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        self.model.load_weights(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        self.model.save_weights(*args, **kwargs)

    @staticmethod
    def build_model(self):
        raise Exception('Not implemented')


class FC1Net(BaseNet):

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
                optimizer='adam',
                metrics=['accuracy'])

        return model
