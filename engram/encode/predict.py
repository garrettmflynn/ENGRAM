import tensorflow as tf
import numpy as np

def predict(model = None,mneme=None):
    channels = len(mneme)
    times = len(mneme[0][0])
    reshape = (-1, channels, times)

    network_input = np.array(mneme).reshape(reshape)
    out = model.predict(network_input)

    choice = np.argmax(out)
    prediction = mneme.options[choice]

    return prediction