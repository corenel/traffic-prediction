from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from utils import data_loader


X_train, y_train = data_loader('data/data302.csv')
print X_train.shape, y_train.shape

in_out_neurons = 1
hidden_neurons = 100
model = Sequential()
# n_prev = 100, 2 values per x axis
model.add(LSTM(hidden_neurons, input_shape=(2952, 5)))
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error",
    optimizer="rmsprop",
    metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=72, nb_epoch=50, validation_split=0.1, verbose=1)
