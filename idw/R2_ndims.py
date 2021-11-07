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
from photutils.utils import ShepardIDWInterpolator as idw


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
flags.DEFINE_integer('neigh', 1000, 'RBF neighbors',
                     short_name='neigh')

def main(argv):
    del argv
######################################################

    ndims = FLAGS.ndims

    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    func = FLAGS.func
    set_scaler = FLAGS.scaler
    load = FLAGS.load
    neighbors = FLAGS.neigh
    low = 0
    high = 1
    alpha = 0.2
    function = Functions(ndims, alpha)

    if load == 0:
        load = False
    elif load == 1:
        load = True


    ##################################################
    r2 = []

    for dim in range(1, ndims+1):
        print("Dimensions = ", dim)
        if func == "D1" or func == "D3" or func == "D6" or func == "D9" or func == "Mxyzuv":
            (_, _, _,
            x_train, y_train, x_test, y_test) = cm.get_functions(func, ndims, alpha, n_train, n_test)

        else:
            (f, f_interp, integral_f) = cm.get_functions(func, dim, alpha)

            x_train = np.random.rand(n_train, dim)
            x_test = np.random.rand(n_test, dim)
            y_test = f(x_test)
            y_train = f(x_train)

        #dim = x_train.shape[1]
        NAME = "{}_d_{}_{:.0e}_sc_{}".format(func, dim, n_train, set_scaler)

        if func == 'Gauss':
            NAME = NAME + '_{:.1f}'.format(alpha)

        print(NAME)
        print("n_train = ", n_train)
        print("set_scaler = ", set_scaler)
        scaler = cm.get_scaler(set_scaler, low = low, high = high, load = load, NAME = NAME)

        y_transf = scaler.fit_transform(y_train.reshape(-1, 1))
        interpolator = idw(x_train, y_transf)
        start = time.time()
        y_pred = np.array(interpolator(x_test)).squeeze()
        end = time.time()
        pred_time = end - start
        y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
        print("IDW Interp done", flush=True)

        r2.append(cm.get_r2(y_test, y_pred))

    np.save("r2_vals_{}".format(NAME), r2)

    plt.figure()
    plt.plot(np.arange(1, ndims+1), r2, label = 'rbf')
    plt.xlabel("Dimensions", fontsize = 'x-large')
    plt.ylabel(r"$R^2$", fontsize = 'x-large')
    plt.savefig("special_plots/R2_vs_ndims_{}.png".format(NAME))
    plt.savefig("special_plots/R2_vs_ndims_{}.pdf".format(NAME))
    plt.legend()



if __name__ == '__main__':
    app.run(main)
