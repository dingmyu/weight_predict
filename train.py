#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor


feature_path = '/root/Desktop/feature/'

if __name__ == '__main__':
 	
	data = np.load(feature_path + 'data.npz')
	X = data['X']#特征
	Y = data['Y']#体重
	min_y = min(Y)
	max_y = max(Y)
	X_train = []
	print min_y
	print max_y
	num_step = 5
	step = (max_y - min_y)/num_step
	for i in range(num_step):#0 1 2 3 4
		idx_temp = np.where(np.logical_and(Y >= min_y + i*step, Y < min_y + (i+1)*step))
		length_temp = idx_temp[0].shape[0]
		train_length = min(length_temp*6/7,length_temp -1)
		if i == 0:
			X_train = X[idx_temp[0][:train_length],:]
			Y_train = Y[idx_temp[0][:train_length]]
			X_test = X[idx_temp[0][train_length:],:]
			Y_test = Y[idx_temp[0][train_length:]]
		else:
			X_train = np.concatenate((X_train,X[idx_temp[0][0:train_length],:]), axis = 0)
			Y_train = np.concatenate((Y_train,Y[idx_temp[0][0:train_length]]), axis = 0)
			X_test = np.concatenate((X_test,X[idx_temp[0][train_length:],:]), axis = 0)
			Y_test = np.concatenate((Y_test,Y[idx_temp[0][train_length:]]), axis = 0)
		print length_temp, train_length
	print X_train.shape
	print Y_train.shape
	print X_test.shape
	print Y_test.shape
	train_length = 640

	clf = RandomForestRegressor(n_estimators = 100, max_features = 11)
	clf.fit(X_train,Y_train)
	Y_pre = clf.predict(X_test)
	score = clf.score(X_test,Y_test)
	print Y_pre.shape
	print type(Y)
	print np.sqrt(((Y_pre - Y_test)**2).mean())#标准差
	print (abs(Y_pre - Y_test)/Y_test).mean()
	total_err=0
	for i in range(Y_pre.shape[0]):
	    err=(Y_pre[i]-Y_test[i])/Y_test[i]
	    total_err+=err*err
	print total_err/Y_pre.shape[0]
	print score
	#print Y_pre - Y_test
	'''error_numpy = Y_pre - Y[train_length:]
	print ((error_numpy - error_numpy.mean())**2).sum'''
	'print score'

	
	YY_pre = clf.predict(X_train)


	plt.figure(figsize=(8, 10))

	plt.subplot(2, 1, 1)
	pl.scatter(np.arange(YY_pre.shape[0]), Y_train, c="b", label="original")
	pl.plot(np.arange(YY_pre.shape[0]), YY_pre, c="r", label="pre function")
	pl.xlabel("X")
	pl.ylabel("Y_pre")
	pl.legend(loc="upper left")
	pl.title("train data")

	plt.subplot(2, 1, 2)
	pl.scatter(np.arange(Y_pre.shape[0]), Y_test, c="b", label="original")
	pl.scatter(np.arange(Y_pre.shape[0]), Y_pre, c="r", label="pre function")
	pl.xlabel("X")
	pl.ylabel("Y_pre")
	pl.legend(loc="upper left")
	pl.title("test data")

	pl.show()
