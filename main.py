from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from random import uniform
from utils import data_loader, train_test_split
# Fix AttributeError: 'module' object has no attribute 'control_flow_ops'
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

if __name__ == '__main__':
    print('-- Loading Data --')
    X, y = data_loader('data/data302.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print('Input shape:', X.shape)
    print('Output shape:', y.shape)

    print('--Creating Model--')
    batch_size = 24
    epochs = 100
    out_neurons = 1
    hidden_neurons = 500
    dropout = uniform(0, 0.5)

    model = Sequential()
    model.add(LSTM(output_dim=hidden_neurons,
                   input_dim=X_train.shape[2],
                   init='uniform',
                   return_sequences=True
                   ))
    model.add(Dropout(dropout))
    model.add(LSTM(output_dim=hidden_neurons,
                   input_dim=hidden_neurons,
                   return_sequences=False
                   ))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=out_neurons, input_dim=hidden_neurons))
    model.add(Activation('relu'))
    model.compile(loss="mse",
                  optimizer="adam",
                  metrics=['accuracy'])

    print('-- Training --')
    history = model.fit(X_train, y_train,
                        verbose=1,
                        batch_size=batch_size,
                        nb_epoch=epochs,
                        validation_split=0.3,
                        shuffle=False
                        )

    print('--Testing--')
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    # print('-- Predicting --')
    # y_pred = model.predict(X_test, batch_size=batch_size)
    #
    # print('-- Plotting Results --')
    # plt.subplot(2, 1, 1)
    # plt.plot(y_test)
    # plt.title('Expected')
    # plt.subplot(2, 1, 2)
    # plt.plot(y_pred)
    # plt.title('Predicted')
    # plt.show()
