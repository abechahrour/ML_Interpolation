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
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
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
from pathlib import Path

import matplotlib as mpl
fpath = Path("~/.fonts/cmr10.ttf")
print(fpath)

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

    NAME = "{}_d_{}_{:.0e}_sc_{}".format(func, ndims, n_train, set_scaler)
    dict_methods = {
        1:'rbf',
        2:'idw',
        3:'nearest',
        4:'grid',
        5:'lgbm',
        6:'svgp',
        7:'nn'
        }
    labels = ['RBF', 'IDW', 'NN', 'Grid', 'LGBM', 'SVGP', 'MLP']
    colors = ['maroon', 'firebrick', 'indianred', 'sienna', 'blue', 'deepskyblue', 'midnightblue']

    r2 = []
    for i in range(len(dict_methods)):
        r2.append(np.load("{}/r2_vals_{}.npy".format(dict_methods[i+1], NAME)))
        if len(r2[i]) == 9:
            r2[i] = r2[i][1:]


    mpl.rcParams['font.family']='serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif']=cmfont.get_name()
    mpl.rcParams['mathtext.fontset']='cm'
    mpl.rcParams['axes.unicode_minus']=False
    plt.figure()
    #csfont = {'fontname':'Comic Sans MS'}
    #hfont = {'fontname':'Book'}
    for i in range(len(dict_methods)):
        print(labels[i])
        y = 1 - r2[i]
        x = np.arange(np.size(y))+2
        plt.plot(x, y, label = labels[i], )
        #plt.text(x[0], y[0], dict_methods[i+1])
    plt.xlabel("Dimensions", fontsize = 'xx-large')
    plt.ylabel(r"$1 - R^2$", fontsize = 'xx-large')
    plt.legend()

    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)
    plt.text(5, 1e-13, 'Periodic', fontsize=14,
        verticalalignment='top', bbox=props)
    #plt.title("Rate of Declining Performance with Increasing Dimension")
    plt.yscale("log")
    #plt.tight_layout()
    plt.savefig("special_plots/1mR2_vs_ndims_{}_solid.png".format(NAME), bbox_inches = 'tight')
    plt.savefig("special_plots/1mR2_vs_ndims_{}_solid.pdf".format(NAME), bbox_inches = 'tight')


    plt.figure()
    #csfont = {'fontname':'Comic Sans MS'}
    #hfont = {'fontname':'Book'}
    for i in range(len(dict_methods)):
        print(labels[i])
        y = r2[i]
        x = np.arange(np.size(y))+2
        plt.plot(x, y, label = labels[i], )
        #plt.text(x[0], y[0], dict_methods[i+1])
    plt.xlabel("Dimensions", fontsize = 'xx-large')
    plt.ylabel(r"$R^2$", fontsize = 'xx-large')
    plt.legend()

    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)
    plt.text(5, 1e-13, 'Periodic', fontsize=14,
        verticalalignment='top', bbox=props)
    #plt.title("Rate of Declining Performance with Increasing Dimension")
    plt.yscale("log")
    #plt.tight_layout()
    plt.savefig("special_plots/R2_vs_ndims_{}_solid.png".format(NAME), bbox_inches = 'tight')
    plt.savefig("special_plots/R2_vs_ndims_{}_solid.pdf".format(NAME), bbox_inches = 'tight')




if __name__ == '__main__':
    app.run(main)
