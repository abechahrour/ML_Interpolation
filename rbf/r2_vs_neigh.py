import numpy as np
import scipy
import pickle
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import sys
sys.path.append('/home/chahrour/Interpolation/')
import sys
from sklearn.metrics import mean_squared_error
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
from tqdm.keras import TqdmCallback
from scipy.interpolate import interpn
from scipy.spatial import Delaunay
import common as cm
import matplotlib.pyplot as plt
from absl import app, flags
import matplotlib
import os
import time
import tensorflow as tf
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
#tf.keras.backend.set_floatx('float32')
FLAGS = flags.FLAGS

flags.DEFINE_string('loss', 'mse', 'The loss function',
                     short_name = 'l')
flags.DEFINE_string('activation', 'elu', 'The Activation',
                     short_name = 'a')
flags.DEFINE_string('dir', 'ti4r4s', 'Directory',
                     short_name = 'dir')
flags.DEFINE_string('opt', 'adam', 'Optimizer',
                     short_name = 'opt')
flags.DEFINE_integer('ndims', 2, 'The number of dimensions',
                     short_name='d')
flags.DEFINE_integer('nout', 1, 'The number of outputs',
                     short_name='nout')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('batch', 100000, 'Number of points to sample per epoch',
                     short_name='b')
flags.DEFINE_integer('nodes', 100, 'Num of nodes',
                     short_name = 'nds')
flags.DEFINE_integer('layers', 8 , 'Exponent of cut function',
                     short_name = 'lyr')
flags.DEFINE_integer('load', 0 , 'Load Model',
                     short_name = 'load')
flags.DEFINE_integer('drop', 0 , 'Dropout',
                     short_name = 'drop')
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
flags.DEFINE_integer('neigh', 1000, 'RBF neighbors',
                     short_name='neigh')




def main(argv):
    del argv
######################################################
    nodes = FLAGS.nodes
    ndims = FLAGS.ndims
    nout = FLAGS.nout
    opt = FLAGS.opt
    layers = FLAGS.layers
    epochs = FLAGS.epochs
    batch = FLAGS.batch
    lr = FLAGS.lr
    loss = FLAGS.loss
    loss_lgbm = 'l2'
    load = FLAGS.load
    real = FLAGS.real
    dir = FLAGS.dir
    func = FLAGS.func
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    activation = FLAGS.activation
    set_scaler = FLAGS.scaler
    drop = FLAGS.drop
    low = 0
    high = 1
    alpha = 0.2
    neighbors = FLAGS.neigh
    set_neighbors = FLAGS.neigh
    #function = Functions(ndims, alpha)


    models = []
    scalers = []
    y_pred = []
    abserr = []
    relerr = []
    perc_err = []
    min_errs = []
    max_errs = []
    mae = []
    rmse = []
    smape = []
    ae_quart_1 = []
    ae_quart_2 = []
    ae_quart_3 = []
    mape = []
    r2 = []
    integrals = []
    stds = []
    pulls = []
    time_pred = []
    r2 = []


    funcs = ['Camel, 9d', 'Periodic, 9d', 'Poly, 3d', 'D3', 'D6', 'D9']
    #set_scaler = ['', '', 'log', '']
    #num_models = [7, 7, 7, 6]
    #color = [colors, colors, colors, colors_6]
    #method = [methods, methods, methods, methods_6]
    #n_test = 1000000
    #set_neighbors = 150

    #colors = ['maroon', 'red', 'tomato', 'blue', 'deepskyblue', 'midnightblue']
    r2 = []
    r2.append(np.load("r2_vals_neigh_Camel_d_9_5e+06_sc_log.npy"))
    r2.append(np.load("r2_vals_neigh_Periodic_d_9_5e+06_sc_.npy"))
    r2.append(np.load("r2_vals_neigh_Poly_d_3_5e+06_sc_.npy"))
    r2.append(np.load("r2_vals_neigh_D3_d_3_5e+06_sc_.npy"))
    r2.append(np.load("r2_vals_neigh_D6_d_6_5e+06_sc_.npy"))
    r2.append(np.load("r2_vals_neigh_D9_d_9_5e+06_sc_.npy"))

    neighbors = np.arange(20, 200, 20)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize = (16, 8))
    #plt.figure()
    counter = 0
    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)
    # place a text box in upper left in axes coords
    for i in range(rows):
        for j in range(cols):
            axes[i, j].plot(neighbors, r2[counter])
            axes[i, j].set_ylabel(r"$R^2$", fontsize = 'x-large')
            axes[i, j].set_xlabel("Neighbors", fontsize = 'x-large')
            #axes[i, j].set_yscale("log")
            axes[i, j].text(0.55, 0.1, funcs[counter], transform=axes[i, j].transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
            counter+=1
    #plt.xticks(ticks = np.arange(1, len(r2[i])+1), labels = labels, fontsize = 'x-large')

    plt.savefig("special_plots/r2_vs_neigh.png", bbox_inches='tight')
    plt.savefig("special_plots/r2_vs_neigh.pdf", bbox_inches='tight')

    # plt.figure()
    # for i in range(len(time)):
    #     plt.bar(np.arange(1, len(time[i])+1), time[i], color = colors, alpha=0.5)
    # plt.xticks(ticks = np.arange(1, len(time[i])+1), labels = labels, fontsize = 'x-large')
    # plt.ylabel("Prediction Time (s)", fontsize = 'xx-large')
    # plt.title("Prediction times on 100k points", fontsize = 'xx-large')
    # plt.yscale("log")
    # plt.savefig("plots/time.png", bbox_inches='tight')
    # plt.savefig("plots_pdf/time.pdf", bbox_inches='tight')
    #

if __name__ == '__main__':
    app.run(main)
