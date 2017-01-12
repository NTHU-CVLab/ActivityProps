from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

class mlp_model():
    
    def __init__():
        # Create model
        model = Sequential()

        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 4096-dimensional vectors.
        model.add(Dense(64, input_dim=4096, init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(64, init='uniform'))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(3, init='uniform'))
        model.add(Activation('softmax'))


        # Compile model
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model
