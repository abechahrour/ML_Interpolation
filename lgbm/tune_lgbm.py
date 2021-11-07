
import sys
sys.path.append('/home/chahrour/Interpolation/')

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import pickle
import pandas as pd

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import vegas
import ndsplines
import lightgbm as lgb
import common as cm
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

from flaml import AutoML
automl = AutoML()

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from scipy.interpolate import griddata, LinearNDInterpolator
from sklearn.metrics import mean_squared_error
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
from tqdm.keras import TqdmCallback
from scipy.interpolate import interpn

from xgboost.callback import LearningRateScheduler
from absl import app, flags
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

tf.keras.backend.set_floatx('float32')
FLAGS = flags.FLAGS

flags.DEFINE_string('loss', 'mse', 'The loss function',
                     short_name = 'l')
flags.DEFINE_string('booster', 'gbtree', 'The Booster',
                     short_name = 'b')
flags.DEFINE_string('dir', 'ti4r4s', 'Directory',
                     short_name = 'dir')

flags.DEFINE_integer('ndims', 2, 'The number of dimensions',
                     short_name='d')
flags.DEFINE_integer('nout', 1, 'The number of outputs',
                     short_name='nout')
flags.DEFINE_integer('n_estimators', 100, 'Number of estimation steps',
                     short_name='n_est')
flags.DEFINE_integer('max_depth', 50, 'Max depth',
                     short_name='max_d')
flags.DEFINE_float('min_child_weight', 1., 'Number of points to sample per epoch',
                     short_name='cw')
flags.DEFINE_float('colsample_bytree', 1., 'Number of points to sample per epoch',
                     short_name='col_samp')
flags.DEFINE_float('subsample', 0.75, 'Subsample',
                     short_name='sub_samp')

flags.DEFINE_integer('load', 0 , 'Load Model',
                     short_name = 'load')
flags.DEFINE_integer('real', 0 , 'Real or Imag',
                     short_name = 'real')
flags.DEFINE_integer('n_train', 5*10**6 , '# of Training Pts',
                     short_name = 'n_train')
flags.DEFINE_integer('n_test', 5*10**6 , '# of Testing Pts',
                     short_name = 'n_test')
flags.DEFINE_float('lr', 1e-3, 'The learning rate',
                     short_name = 'lr')
flags.DEFINE_string('func', 'Gauss', 'Function',
                     short_name = 'f')





class Functions:


    def __init__(self, ndims, alpha, **kwargs):
        self.ndims = ndims
        self.alpha = alpha
        self.variables = kwargs
        self.calls = 0

    def gauss(self, x):

        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.0*np.sum(((x-0.5)**2)/self.alpha**2, axis=-1)
        #self.calls += 1
        return pre * np.exp(exponent)
    @vegas.batchintegrand
    def poly(self, x):
        #print("Shape of poly = ", np.shape(x))
        # res = 0
        # for d in range(np.shape(x)[1]):
        #     res += -x[:,d]**2+x[:,d]
        # return res
        return np.sum(-x**2 + x, axis=-1)


    def periodic(self, x):
        return np.mean(x, axis = -1)*np.prod(np.sin(2*np.pi*x), axis = -1)
    def D0(self, x):
        path = '/home/chahrour/Loops/D0000stmmmm/D0000stmmmm_data/D00001rssss_labels_5M.csv'


def poly_interp(x):
    return np.sum(-x**2 + x, axis=0)

def mag(x):
    return 10**(np.floor(np.log10(np.abs(x))))


def main(argv):
    del argv
    # load data

    ndims = FLAGS.ndims
    nout = FLAGS.nout
    n_estimators = FLAGS.n_estimators
    max_depth = FLAGS.max_depth
    lr = FLAGS.lr
    min_child_weight = FLAGS.min_child_weight
    subsample = FLAGS.subsample
    colsample_bytree = FLAGS.colsample_bytree
    booster = FLAGS.booster
    loss = FLAGS.loss
    load = FLAGS.load
    real = FLAGS.real
    func = FLAGS.func
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    set_scaler = ""
    alpha = 0.2
    dir = '/home/chahrour/Interpolation/lgbm/'
    function = Functions(ndims, alpha)
    if func == 'Gauss':
        f = function.gauss
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
        idcs = np.where(y_train > 1e-8)
        x_train = x_train[idcs]
        y_train = y_train[idcs]
        if (alpha < 0.4 and ndims > 3):
            set_scaler = "log"

    elif func == 'Periodic':
        f = function.periodic
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)

    elif func == 'Poly':
        f = function.poly
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)

    elif func =='D0':
        if ndims != 2:
            print("ndims should equal 2!")
            exit()
        f = function.D0
        path_feat = '/home/chahrour/Loops/D0000stmmmm/D0000stmmmm_data/D00001rssss_features_5M.csv'
        path_labels = '/home/chahrour/Loops/D0000stmmmm/D0000stmmmm_data/D00001rssss_labels_5M.csv'
        x_train = np.array(pd.read_csv(path_feat, delimiter=',', nrows = n_train))
        y_train = np.array(pd.read_csv(path_labels, delimiter=',', nrows = n_train))[:,0]
        y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = np.array(pd.read_csv(path_feat, delimiter=',', nrows = n_test))
        y_test = np.array(pd.read_csv(path_labels, delimiter=',', nrows = n_test))[:,0]
        y_test = y_test * x_test[:,0] * x_test[:,1]**2


    if load == 0:
        load = False
    elif load == 1:
        load = True


    NAME = "LGBM_{}_n_{}_md_{}_cw_{}_ss_{}_cs_{}_l_{}_d_{}_b_{}_{}_{:.0e}".format(func, n_estimators,
            max_depth, min_child_weight, subsample, colsample_bytree, loss, ndims,
            booster,set_scaler, n_train)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    print(NAME)

    if load:
        scaler = pickle.load(open(dir+'scaler_lgbm/scaler_{}'.format(NAME), 'rb'))
    else:
        if set_scaler == "ss":
            scaler = StandardScaler()
        elif set_scaler == "mm":
            scaler = MinMaxScaler((low, high))
        elif set_scaler == "log":
            scaler = FunctionTransformer(cm.log_transform, inverse_func = cm.exp_transform)
        else:
            scaler = FunctionTransformer()

    ########## Scale #########################
    y_train = (scaler.fit_transform(y_train.reshape(-1, 1))).ravel()

    ##################################################
    settings = {
    "time_budget": 60*60*4,  # total running time in seconds
    "metric": 'mae',  # primary metrics for regression can be chosen from: ['mae','mse','r2']
    "estimator_list": ['lgbm'],  # list of ML learners; we tune lightgbm in this example
    "task": 'regression',  # task type
    "log_file_name": 'houses_experiment.log',  # flaml log file
    "seed": 7654321,    # random seed
    }
    automl.fit(X_train=x_train, y_train=y_train, **settings)
    ''' retrieve best config'''
    print('Best hyperparmeter config:', automl.best_config)
    print('Best r2 on validation data: {0:.4g}'.format(1-automl.best_loss))
    print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
    automl.model.estimator


if __name__ == '__main__':
    app.run(main)
