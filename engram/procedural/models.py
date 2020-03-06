from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization


def select(name):
    selection = {
        "CNN": cnn,
        'MIMO': mimo,
    }
    # Get the function from switcher dictionary
    func = selection.get(name, lambda: "Invalid model")
    # Execute the function
    return func()




def mimo():
    print('in development')

def cnn(train_X):
    model = Sequential()

    model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
    model.add(Activation('relu'))

    model.add(Conv1D(64, (2)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(64, (2)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2), padding='same'))

    model.add(Flatten())

    model.add(Dense(512))

    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])  

    return model