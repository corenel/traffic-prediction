import csv
import numpy as np
from datetime import datetime

def data_loader(filename):
	X = []
	Y = []
	with open('data302.csv') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			date = datetime.strptime(row['Time'], '%Y-%m-%d  %H:%M:%S')
			X.append([date.hour, row['Route'], row['Weather'], row['Roadtype'], row['Holiday']])
			Y.append([row['Velocity']])
	X = np.array(X)
	Y = np.array(Y)
	return X, Y

def split_train_test_data(X, Y, num_train):


if __name__ == "__main__":
	X, Y = data_loader('data302.csv')
	print X.shape, Y.shape