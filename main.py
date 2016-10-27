from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from utils import data_loader, train_test_split

if __name__ == '__main__':
    # since we are using stateful rnn tsteps can be set to 1
    tsteps = 1
    batch_size = 24
    epochs = 50
    # number of elements ahead that are used to make the prediction
    lahead = 1

    print('-- Loading Data --')
    X, y = data_loader('data/data302.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y)
    print('Input shape:', X.shape)
    print('Output shape:', y.shape)

    print('--Creating Model--')
    in_out_neurons = 1
    hidden_neurons = 500

    model = Sequential()
    model.add(LSTM(hidden_neurons, input_shape=(5, 1)))
    model.add(Dense(in_out_neurons))
    model.add(Activation("linear"))

    model.compile(loss="mean_squared_error",
        optimizer="adam",
        metrics=['accuracy'])

    print('-- Training --')
    model.fit(X_train, y_train, batch_size=24, nb_epoch=50, validation_data=(X_test, y_test), verbose=1)
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
