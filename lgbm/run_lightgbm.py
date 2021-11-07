
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
import collections
from operator import gt, lt
from typing import Any, Callable, Dict, List, Union
from Functions import Functions
#from .basic import _ConfigAliases, _log_info, _log_warning

from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer


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
flags.DEFINE_integer('num_leaves', 100, 'Number of leaves',
                     short_name='n_leaf')
flags.DEFINE_integer('max_depth', 50, 'Max depth',
                     short_name='max_d')
flags.DEFINE_integer('max_bin', 50, 'Max depth',
                     short_name='max_b')
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
flags.DEFINE_string('scaler', '', 'Scaling/Normalization',
                     short_name = 's')


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
    bagging_fraction = FLAGS.subsample
    feature_fraction = FLAGS.colsample_bytree
    booster = FLAGS.booster
    loss = FLAGS.loss
    load = FLAGS.load
    real = FLAGS.real
    func = FLAGS.func
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    num_leaves = FLAGS.num_leaves
    max_bin = FLAGS.max_bin
    set_scaler = FLAGS.scaler
    alpha = 0.2
    dir = '/home/chahrour/Interpolation/lgbm/'
    function = Functions(ndims, alpha)
    low = 0
    high = 1

    if load == 0:
        load = False
    elif load == 1:
        load = True



    if func == "D3" or func == "D6" or func == "D9" or func == "Mxyzuv":
        _, _, _, x_train, y_train, x_test, y_test = cm.get_functions(func, ndims, alpha, n_train, n_test)
    else:
        (f, f_interp, integral_f) = cm.get_functions(func, ndims, alpha, n_train, n_test)
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
    ndims = x_train.shape[1]

    NAME = "LGBM_{}_d_{}_n_{}_md_{}_mb_{}_nleaf_{}_ss_{}_cs_{}_l_{}_b_{}_lr_{}_{}_{:.0e}".format(func, ndims, n_estimators,
            max_depth, max_bin, num_leaves, bagging_fraction, feature_fraction, loss,
            booster, lr, set_scaler, n_train)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    print(NAME)
    scaler = cm.get_scaler(set_scaler = set_scaler, low = low, high = high, load = load, NAME = NAME)

    ########## Scale #########################
    y_train = (scaler.fit_transform(y_train.reshape(-1, 1))).ravel()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                        test_size=0.1, random_state=1) # 0.25 x 0.8 = 0.2
    ##################################################

    print("x_train = ", x_train)
    print("y_train = ", y_train)
    print("x_test = ", x_test)
    print("y_test = ", y_test)
    print("y_train max = ", np.max(y_train))
    print("y_test max = ", np.max(y_test))

    if load:
        #model = lgb.Booster(model_file=dir+"models_lgbm/{}".format(NAME))
        model = pickle.load(open('models/{}'.format(NAME), 'rb'))
        scaler = pickle.load(open('scaler/scaler_{}'.format(NAME), 'rb'))
        print("Model has loaded")
    else:
        params = {
        # Parameters that we are going to tune.
        'n_estimators':n_estimators,
        'max_depth':max_depth,
        'learning_rate':lr,
        'bagging_fraction': bagging_fraction,
        'colsample_bytree': feature_fraction,
        'boosting_type':booster,
        'objective':loss,
        'num_leaves':num_leaves,
        #'min_child_samples': 298,
        #'bagging': 0.5,
        'max_bin': max_bin
        }
        def learning_rate_005_decay_power_099(current_iter):
            base_learning_rate = lr
            learning_r = base_learning_rate  * np.power(.9999, current_iter)
            if current_iter % 100 == 0:
                print(learning_r)
            return learning_r if learning_r > 1e-3 else 1e-3
        print_callback = lgb.print_evaluation(100, True)
        learning_rates = []
        lr_temp = lr
        for i in range(n_estimators):
            if i % 5 == 0:
                lr_temp = lr_temp*0.98
                learning_rates.append(lr_temp)
            else:
                learning_rates.append(lr_temp)
        #print("lrs = ", learning_rates)
        lr_schedule = LearningRateScheduler(learning_rates)
        models = []
        y_pred = []

        if tf.test.is_gpu_available():
            model = LGBMRegressor(**params, tree_method = 'gpu_hist')
        else:
            model = LGBMRegressor(**params)
        if booster == 'gbdt':
            model.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_val, y_val)],
                     eval_metric=loss, early_stopping_rounds=1000,
                     verbose=False,
                     callbacks=[print_callback,
                     lgb.reset_parameter(learning_rate = learning_rate_005_decay_power_099)])
        elif booster == 'dart':
            model.fit(x_train, y_train, eval_set=[(x_train, y_train),(x_val, y_val)],
                     eval_metric=loss, early_stopping_rounds=1000,
                     verbose=False,
                     callbacks=[print_callback])
            #model.booster_.save_model(dir+"models_lgbm/{}".format(NAME))
        pickle.dump(model, open('models/{}'.format(NAME), 'wb'))
        pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))

    start = time.time()
    y_pred = model.predict(x_test)
    end = time.time()
    print("Time taken for prediction = ", end - start, " sec")
    y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()


    # evaluate the model
    # cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
    # n_scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # print(n_scores)
    #predictions = [round(value) for value in y_pred]


    # evaluate predictions
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: {:.2f}".format(mse))


    #############################    INTEGRATION     ##########################

    int_naive, std_naive = cm.get_integral_uniform(y_test)
    integral, std = cm.get_integral_uniform(y_pred)

    try:
        int_actual = integral_f(ndims, alpha)
        print("The True Integral = ", int_actual)

        integ = vegas.Integrator(ndims* [[0, 1]])
        int_vegas = integ(f, nitn=10, neval=50000)

        print(int_vegas.summary())
        print('int_vegas = %s    Q = %.2f' % (int_vegas, int_vegas.Q))
        pull = cm.pull(int_actual, integral,std)
    except Exception as e:
        print("Exception tossed: ", e)
        #print("f is not defined")
        int_vegas = 0
        pull = cm.pull(int_naive, integral,std)
    print('Naive Integral = {} +- {}'.format(int_naive, std_naive))
    print('Predicted Integral = {} +- {}'.format(integral, std))
    print("pull = {}".format(pull))


    ############ Compute Measures ###############################

    N = 5*10**4
    log_acc = cm.get_logacc(y_test, y_pred)
    relerr = cm.get_relerr(y_test, y_pred)
    mape = np.mean(np.abs(relerr))
    abserr = cm.get_abserr(y_test, y_pred)
    err = cm.get_err(y_test, y_pred)
    mse = cm.get_mse(y_test, y_pred)
    mae = np.mean(abserr)
    sigfigs = cm.get_sigfigs(y_test, y_pred)
    r2 = cm.get_r2(y_test, y_pred)

    pts = []
    eta = []
    for i in range(8):
        pts.append((np.logical_and(sigfigs < -i, sigfigs > -(i+1))).sum())
        eta.append(pts[i]/sigfigs.size*100)


    #########################################################

    measures = {
        'mse':mse,
        'mae':mae,
        'err':err,
        'sigfigs':sigfigs,
        'relerr':relerr,
        'mape':mape,
        'func':func,
        'model':model,
        'log_acc':log_acc,
        'eta':eta,
        'alpha':alpha,
        'loss':loss
    }

    csv_output = {
        'func':     func,
        'ndims':    [ndims],
        'n_estimators':    [n_estimators],
        'max_depth':   [max_depth],
        'min_child_weight':[min_child_weight],
        'subsample':      [bagging_fraction],
        'colsample_bytree': feature_fraction,
        'loss':     loss,
        'booster':  booster,
        'set_scaler':set_scaler,
        'n_train':  [n_train],
        'n_test':   [n_test],
        'mse':      [mse],
        'mae':      [mae],
        'mape':     [mape],
        'integral':   [integral],
        'std':   [std],
        'int_naive':[int_naive],
        'int_vegas':[int_vegas],
        'pull':     [pull],
        'datetime': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        'r2':   [r2],
        'lr':   [lr]
    }

    pd.DataFrame.from_dict(data=csv_output).to_csv(dir+'lgbm_runs.csv', mode='a', header = False, index = False)

    #########################################################

    cm.plot_lgbm(y_test, y_pred, measures, 'LGBM', NAME)


if __name__ == '__main__':
    app.run(main)
