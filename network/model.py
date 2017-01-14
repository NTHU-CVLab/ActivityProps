from keras.models import Sequential
from keras.layers import Dense, Dropout


class FC4Net:

    def __init__(self):
        self.model = self.build_model()

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=4096, init='uniform', activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def compile_model(self):
        self.model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
