import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

feature_path = '/root/Desktop/feature/'

if __name__ == '__main__':
	gbdt=GradientBoostingRegressor(n_estimators = 10, max_depth = 5, learning_rate = 0.1, loss = 'lad', max_features = 5)
	data = np.load(feature_path + 'data_1.npz')
	X = data['X']
	Y = data['Y']
	min_y = min(Y)
	max_y = max(Y)
	print min_y
	print max_y
	num_step = 5
	step = (max_y - min_y)/num_step
	for i in range(num_step):
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
	gbdt.fit(X_train,Y_train)
	pred=gbdt.predict(X_test)
	score = gbdt.score(X_test,Y_test)
	total_err=0
	pred = pred[6:-6]
	Y_test = Y_test[6:-6]
	print pred - Y_test
	print np.where(abs(pred-Y_test)>500)
	print (abs(pred - Y_test)/Y_test).mean()
	for i in range(pred.shape[0]):
	    err=(pred[i]-Y_test[i])/Y_test[i]
	    total_err+=err*err
	print total_err/pred.shape[0]
	print score
	
	YY_pre = gbdt.predict(X_train)
	
	sort_idx = Y_test.argsort()
	Y_testsort = Y_test[sort_idx]
	Y_presort = pred[sort_idx]

	sort_idx = Y_train.argsort()
	Y_trainsort = Y_train[sort_idx]
	YY_presort = YY_pre[sort_idx]

	plt.figure(figsize=(8, 10))

	plt.subplot(2, 2, 1)
	pl.scatter(np.arange(YY_pre.shape[0]), Y_train, c="b", label="original")
	pl.plot(np.arange(YY_pre.shape[0]), YY_pre, c="r", label="pre function")
	pl.xlabel("X")
	pl.ylabel("Y_pre")
	pl.legend(loc="upper left")
	pl.title("train data")

	plt.subplot(2, 2, 2)
	pl.scatter(np.arange(pred.shape[0]), Y_test, c="b", label="original")
	pl.scatter(np.arange(pred.shape[0]), pred, c="r", label="pre function")
	pl.xlabel("X")
	pl.ylabel("Y_pre")
	pl.legend(loc="upper left")
	pl.title("test data")

	plt.subplot(2, 2, 3)
	pl.scatter(np.arange(YY_pre.shape[0]), Y_trainsort, c="b", label="original")
	pl.plot(np.arange(YY_pre.shape[0]), YY_presort, c="r", label="pre function")
	pl.xlabel("X")
	pl.ylabel("Y_pre")
	pl.legend(loc="upper left")
	pl.title("train data")

	plt.subplot(2, 2, 4)
	pl.scatter(np.arange(pred.shape[0]), Y_testsort, c="b", label="original")
	pl.scatter(np.arange(pred.shape[0]), Y_presort, c="r", label="pre function")
	pl.xlabel("X")
	pl.ylabel("Y_pre")
	pl.legend(loc="upper left")
	pl.title("test data")

	pl.show()
