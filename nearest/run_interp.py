import tensorflow as tf

import sys
sys.path.append('/home/chahrour/Interpolation/')
import pickle

import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import vegas
import ndsplines
import pandas as pd
from datetime import datetime
import common as cm
from Functions import Functions



from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels

from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator

from sklearn.metrics import mean_squared_error
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
from tqdm.keras import TqdmCallback
from scipy.interpolate import interpn
from scipy.spatial import Delaunay


from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer


from absl import app, flags


tf.keras.backend.set_floatx('float32')
FLAGS = flags.FLAGS

flags.DEFINE_string('loss', 'mse', 'The loss function',
                     short_name = 'l')

flags.DEFINE_string('dir', 'ti4r4s', 'Directory',
                     short_name = 'dir')

flags.DEFINE_integer('ndims', 2, 'The number of dimensions',
                     short_name='d')
flags.DEFINE_integer('nout', 1, 'The number of outputs',
                     short_name='nout')
flags.DEFINE_integer('load', 0 , 'Load Model',
                     short_name = 'load')
flags.DEFINE_integer('real', 0 , 'Real or Imag',
                     short_name = 'real')
flags.DEFINE_integer('n_train', 5*10**6 , '# of Training Pts',
                     short_name = 'n_train')
flags.DEFINE_integer('n_rbf', 10**4 , '# of Training Pts for RBF',
                     short_name = 'n_rbf')
flags.DEFINE_integer('n_interp', 10**4 , '# of Training Pts for Linear Interp',
                     short_name = 'n_interp')
flags.DEFINE_integer('n_lin', 10**4 , '# of Training Pts for Linear Interp',
                     short_name = 'n_lin')
flags.DEFINE_integer('n_grid', 10**4 , '# of Training Pts for Grid Interp',
                     short_name = 'n_grid')
flags.DEFINE_integer('n_test', 5*10**6 , '# of Testing Pts',
                     short_name = 'n_test')
flags.DEFINE_float('lr', 1e-3, 'The learning rate',
                     short_name = 'lr')
flags.DEFINE_string('func', 'Gauss', 'Function',
                     short_name = 'f')
flags.DEFINE_string('scaler', '', 'Scaling/Normalization',
                     short_name = 's')

def main(argv):
    del argv
######################################################

    ndims = FLAGS.ndims

    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    func = FLAGS.func
    set_scaler = FLAGS.scaler
    load = FLAGS.load
    neighbors = 1000
    low = 0
    high = 1
    alpha = 0.2
    function = Functions(ndims, alpha)

    if load == 0:
        load = False
    elif load == 1:
        load = True


    ##################################################

    if func == "D1" or func == "D3" or func == "D6" or func == "D9" or func == "Mxyzuv":
        (_, _, _, x_train, y_train, x_test, y_test) = cm.get_functions(func, ndims, alpha, n_train, n_test)

        #y_interp_grid = np.ones_like(y_rbf)
        #y_interp = np.ones_like(y_rbf)
    else:
        (f, f_interp, integral_f) = cm.get_functions(func, ndims, alpha)


        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
    ndims = x_train.shape[1]
    NAME = "{}_d_{}_{:.0e}_sc_{}".format(func, ndims, n_train, set_scaler)

    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)

    print(NAME)
    print("n_train = ", n_train)
    scaler = cm.get_scaler(set_scaler, low = low, high = high, load = load, NAME = NAME)

    y_transf = scaler.fit_transform(y_train.reshape(-1, 1))
    interpolator = NearestNDInterpolator(x_train, y_transf)
    start = time.time()
    y_pred = np.array(interpolator(x_test)).squeeze()
    end = time.time()
    pred_time = end - start
    y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
    print("Nearest Neighbor Interp done", flush=True)
    ################# Save Interpolants #######################

    pickle.dump(interpolator, open('interpolants/{}'.format(NAME), 'wb'))
    pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
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
    # MAPE
    #mape = ((y_test - y_pred)/(y_test) * 100)

    log_acc = []
    relerr = []
    mape = []
    abserr = []
    err = []
    mse = []
    mae = []
    sigfigs = []
    r2 = []

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





    #####################################################
    measures = {
        'mse':mse,
        'mae':mae,
        'err':err,
        'sigfigs':sigfigs,
        'relerr':relerr,
        'mape':mape,
        'func':func,
        'log_acc':log_acc,
        'eta':eta,
        'alpha':alpha,
    }

    csv_output = {
        'func':     func,
        'ndims':    [ndims],
        'set_scaler':set_scaler,
        'n_train':  [n_train],
        'n_test':   [n_test],
        'mse':      [mse],
        'mae':      [mae],
        'mape':     [mape],
        'int':   [integral],
        'err':   [std],
        'int_naive':[int_naive],
        'int_vegas':[int_vegas],
        'pull':     [pull],
        'r2':   [r2],
        'time': [pred_time],
        'datetime': datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    }

    pd.DataFrame.from_dict(data=csv_output).to_csv('interp_runs.csv', mode='a', header = False, index = False)

    #######################################################

    cm.plot_nearest(y_test, y_pred, measures, 'nearest', NAME)

if __name__ == '__main__':
    app.run(main)
