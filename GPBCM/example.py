from GPyBCM import BCM
import numpy as np
from matplotlib import pyplot as plt

def periodic(x):
	return np.mean(x, axis = -1)*np.prod(np.sin(2*np.pi*x), axis = -1)

if __name__ == '__main__':

	N = 100000
	x_train = np.random.rand(N,5) # 5 features
	y_train = periodic(x_train)

	bcm = BCM(x_train, np.array(y_train).reshape(-1,1), M=40,N=300,verbose=2) # default model: rBCM

	print('Optimizing hyperparameters..')
	bcm.optimize() # optimize sum of log-likelihood of experts

	print(bcm.param_array)

	x_test = np.random.rand(N,5)
	y_test = periodic(x_train)

	y_pred_rbcm = bcm.predict(x_test)
	bcm.model = 'gpoe' # generalized product of experts
	y_pred_gpoe = bcm.predict(x_test)

	print('rBCM : ' + str(np.linalg.norm(y_test - y_pred_rbcm)))
	print('GPoE : ' + str(np.linalg.norm(y_test - y_pred_gpoe)))

	print(y_pred_rbcm.shape)
	print(y_test.shape)
	y_pred_rbcm = y_pred_rbcm.squeeze()

	err = y_test - y_pred_rbcm
	relerr = err/y_test * 100

	plt.hist(relerr, range = (-100, 100), bins = 100)
	plt.yscale('log')
	plt.savefig("relerr.png")
