from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.utils.visualize_util import plot
from random import uniform
from datetime import datetime
from utils import data_loader, train_test_split
import json
# Fix AttributeError: 'module' object has no attribute 'control_flow_ops'
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

if __name__ == '__main__':
    print('-- Loading Data --')
    test_size = 1728
    X, y = data_loader('data/data_pems_16664.csv')
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size)
    print('Input shape:', X.shape)
    print('Output shape:', y.shape)

    print('-- Reading pre-trained model and weights --')
    with open('model/model_3_layer.json') as f:
        json_string = json.load(f)
        model = model_from_json(json_string)
    model.load_weights('model/weights_3_layer.h5')

    # print('-- Creating Model--')
    batch_size = 96
    # epochs = 100
    # out_neurons = 1
    # hidden_neurons = 500
    # hidden_inner_factor = uniform(0.1, 1.1)
    # hidden_neurons_inner = int(hidden_inner_factor * hidden_neurons)
    # dropout = uniform(0, 0.5)
    # dropout_inner = uniform(0, 1)
    #
    # model = Sequential()
    # model.add(LSTM(output_dim=hidden_neurons,
    #                input_dim=X_train.shape[2],
    #                init='uniform',
    #                return_sequences=True,
    #                consume_less='mem'))
    # model.add(Dropout(dropout))
    # model.add(LSTM(output_dim=hidden_neurons_inner,
    #                input_dim=hidden_neurons,
    #                return_sequences=True,
    #                consume_less='mem'))
    # model.add(Dropout(dropout_inner))
    # model.add(LSTM(output_dim=hidden_neurons_inner,
    #                input_dim=hidden_neurons_inner,
    #                return_sequences=False,
    #                consume_less='mem'))
    # model.add(Dropout(dropout_inner))
    # model.add(Activation('relu'))
    # model.add(Dense(output_dim=out_neurons,
    #                 input_dim=hidden_neurons_inner))
    # model.add(Activation('relu'))
    model.compile(loss="mse",
                  optimizer="adam",
                  metrics=['accuracy'])
    #
    #
    # print('-- Training --')
    # history = model.fit(X_train,
    #                     y_train,
    #                     verbose=1,
    #                     batch_size=batch_size,
    #                     nb_epoch=epochs,
    #                     validation_split=0.1,
    #                     shuffle=False)

    print('-- Evaluating --')
    eval_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    print('Evaluate loss: ', eval_loss[0])
    print('Evaluate accuracy: ', eval_loss[1])

    print('-- Predicting --')
    y_pred = model.predict(X_test, batch_size=batch_size)

    print('-- Plotting Results --')
    plt.style.use('ggplot')
    plt.plot(y_test, label='Expected', linewidth=2)
    plt.plot(y_pred, label='Predicted')
    plt.title('Traffic Prediction')
    plt.xlabel('Smaple')
    plt.ylabel('Velocity')
    plt.xlim(0, test_size)
    plt.legend()
    plt.show()


    print('-- Saving results --')
    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    pd.DataFrame(y_pred).to_csv('predict/y_pred_' + now + '.csv')
    pd.DataFrame(y_test).to_csv('predict/y_test_' + now + '.csv')
    with open('model/model_' + now + '.json', 'w') as f:
        json.dump(model.to_json(), f)
    model.save_weights('model/weights_' + now + '.h5', overwrite=True)
    Model Visualization
    plot(model, to_file='img/model.png', show_shapes=True)
