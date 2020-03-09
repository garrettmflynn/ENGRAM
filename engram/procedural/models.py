from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization


def select(model,shape):
    selection = {
        "CNN": cnn,
        'MIMO': mimo,
        "LSTM":lstm,
        'custom':custom
    }
    # Get the function from switcher dictionary
    func = selection.get(model, lambda: "Invalid model")
    # Execute the function
    return func(shape)

def mimo(shape):
    print('in development')

def lstm():
    print(shape)
    model = Sequential()
    model.add(LSTM(32,input_shape=shape))
    model.add(Dense(1))

# INPUT DIMENSION MEANINGS
# 1. Samples. One sequence is one sample. A batch is comprised of one or more samples.
# 2. Time Steps. One time step is one point of observation in the sample.
# 3. Features. One feature is one observation at a time step.

def cnn(shape):

    model = Sequential()

    model.add(Conv1D(64, (3), input_shape=shape))
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

def custom(shape):
    import custommodelconfig

    model = tf.keras.models.model_from_config(
    custommodelconfig, custom_objects=None
    )

    return model

