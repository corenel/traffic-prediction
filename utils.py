import csv
import numpy as np
from datetime import datetime

def data_loader(filename):
	X = []
	y = []
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			# date = datetime.strptime(row['Time'], '%Y-%m-%d  %H:%M:%S')
			# X.append([date.hour, row['Weather'], row['Holiday']])
			# y.append([row['Velocity']])
			date = datetime.strptime(row['5 Minutes'], '%Y-%m-%d  %H:%M:%S')
			X.append([date.weekday(), is_weekend(date), date.hour, date.minute])
			y.append([row['Mainline Agg']])
	X = np.array(X)
	X = X.reshape(X.shape[0], 1, X.shape[1])
	y = np.array(y)
	return X, y

def train_test_split(X, y, test_size=288):
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    split_idx = X.shape[0] - test_size
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def is_weekend(date):
	return int(date.weekday() in [5, 6])
