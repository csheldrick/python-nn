"""
The design of this comes from here:
http://outlace.com/Reinforcement-Learning-Part-3/
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.optimizers import RMSprop, Adam
from keras.layers.recurrent import LSTM
from keras.callbacks import Callback

# Adding this per a suggestion by Tim Kelch.
# https://medium.com/@trkelch/this-post-is-great-possibly-the-best-tutorial-explanation-ive-found-thus-far-cf78886b5378#.w473ywtbw
import tensorflow as tf
# tf.python.control_flow_ops = tf


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def neural_net(num_sensors, params, load=''):
    model = Sequential()
    # First layer.
    model.add(Dense(params[0], input_shape=(num_sensors,), activation='relu'))
    #model.add(Dropout(0.2))
    # Second layer.
    model.add(Dense(params[1], activation='relu'))
    #model.add(Dropout(0.2))
    # Output layer.
    model.add(Dense(4, activation='linear'))

    # rms = RMSprop()
    adam = Adam(lr=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=['mse','accuracy'])

    if load:
        model.load_weights(load)

    return model


def lstm_net(num_sensors, load=False):
    model = Sequential()
    model.add(LSTM(
        output_dim=512, input_dim=num_sensors, return_sequences=True
    ))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim=512, input_dim=512, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=3, input_dim=512))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")

    return model