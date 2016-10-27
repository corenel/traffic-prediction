import csv
import numpy as np
from datetime import datetime

def data_loader(filename):
	X = []
	y = []
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			date = datetime.strptime(row['Time'], '%Y-%m-%d  %H:%M:%S')
			X.append([date.hour, row['Route'], row['Weather'], row['Roadtype'], row['Holiday']])
			y.append([row['Velocity']])
	X = np.array(X)
	y = np.array(y)
	return X, y

# def split_train_test_data(X, Y, num_train):
