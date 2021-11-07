import numpy as np
import scipy
import pickle
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

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
    ram = []
    methods = {
        1:'rbf',
        2:'idw',
        3:'nearest',
        4:'grid',
        5:'lgbm',
        6:'svgp',
        7:'nn'
        }
    methods_6 = {
        1:'rbf',
        2:'idw',
        3:'nearest',
        4:'lgbm',
        5:'svgp',
        6:'nn'
        }
    labels = ['RBF', 'IDW', 'NN', 'Grid', 'LGBM', 'SVGP', 'MLP']
    labels_6 = ['RBF', 'IDW', 'NN', 'LGBM', 'SVGP', 'MLP']
    label = [labels, labels, labels, labels_6]
    colors = ['maroon', 'firebrick', 'indianred', 'sienna', 'blue', 'deepskyblue', 'midnightblue']
    #colors = ['blue', 'blue', 'blue', 'blue', 'red', 'red', 'red']
    colors_6 = ['maroon', 'firebrick', 'indianred', 'blue', 'deepskyblue', 'midnightblue']
    func = 'Mxyzuv'
    set_scaler = ''
    num_models = [7, 7, 7, 6]
    color = [colors, colors, colors, colors_6]
    method = [methods, methods, methods, methods_6]
    n_train = 4000000
    n_test = 1000000
    set_neighbors = 150

    #colors = ['maroon', 'red', 'tomato', 'blue', 'deepskyblue', 'midnightblue']

    y_pred = []
    y_test = []

    NAME = "{}_d_{}_{:.0e}_sc_{}".format(func, ndims, n_train, set_scaler)
    NAME_test = NAME +'_{:.0e}_neigh_{}'.format(n_test, set_neighbors)

    print(NAME_test)
    y_test.append(np.load("predictions/y_test_{}.npy".format(NAME_test)))
    for i in range(num_models[3]):

        y_pred.append(np.load("predictions/y_pred_{}_{}.npy".format(method[3][i+1], NAME_test)))


    print("Predicting values of models\n Computing metrics", flush = True)


    counter = 0
    y_test = np.array(y_test).squeeze()
    print("y_pred = ", np.shape(y_pred))
    print("y_test = ", np.shape(y_test))


    #print(funcs[j])
    for i in range(num_models[3]):
        print(np.shape(y_test))
        print(np.shape(y_pred[counter]))
        smape.append(cm.get_smape(y_test, y_pred[counter]))
        relerr.append(cm.get_relerr(y_test, y_pred[counter]))
        abserr.append(cm.get_abserr(y_test, y_pred[counter]))
        r2.append(cm.get_r2(y_test, y_pred[counter]))
        counter += 1
        print(counter)

    print(np.shape(abserr))
    print("r2 = ", np.shape(r2))
    rows = 1
    cols = 3
    textstr = 'M'
    fig, axs = plt.subplots(rows, cols, figsize = (14, 4))
    counter = 0
    count_func = 0
    #meanlineprops = dict(linestyle='-', linewidth = 1, color='gold')
    meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='lime')
    medianlineprops = dict(linestyle='-', linewidth = 1)
    flierprops = dict(marker='o', markerfacecolor='black', markersize=4,
                  markeredgecolor='none')

    bplot = axs[0].boxplot(np.transpose(abserr),
                              meanprops = meanpointprops,
                              medianprops=medianlineprops,
                              flierprops = flierprops,
                              meanline=False,
                              showmeans=True,
                              showfliers = False,
                              widths = 0.5,
                              notch=False,  # notch shape
                              vert=True,   # vertical box aligmnent
                              patch_artist=True)   # fill with color
    for patch, color in zip(bplot['boxes'], colors_6):
        patch.set_facecolor(color)

    print("Colors Set")
    axs[0].yaxis.grid(True)
    axs[0].set_xticks([y+1 for y in range(num_models[3])])
    axs[0].set_xticklabels(labels_6, fontsize = 'x-large')
    axs[0].tick_params(axis='y', which='major', labelsize=12)
    #ax1.set_xlabel('Models', fontsize = 'xx-large')
    axs[0].set_ylabel('Absolute Error', fontsize = 'xx-large')
    axs[0].set_yscale('log')
    nticks = 30
    maj_loc = matplotlib.ticker.LogLocator(numticks=nticks)
    min_loc = matplotlib.ticker.LogLocator(subs='all', numticks=nticks)
    axs[0].yaxis.set_major_locator(maj_loc)
    axs[0].yaxis.set_minor_locator(min_loc)
    axs[0].set_ylim(bottom = np.min(np.quantile(np.array(abserr), 0.05, axis = 1)))
    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)

    # place a text box in upper left in axes coords
    axs[0].text(0.45, 0.1, textstr, transform=axs[0].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    bplot2 = axs[1].boxplot(np.transpose(smape),
                              meanprops = meanpointprops,
                              medianprops=medianlineprops,
                              flierprops = flierprops,
                              meanline=False,
                              showmeans=True,
                              showfliers = False,
                              widths = 0.5,
                              notch=False,  # notch shape
                              vert=True,   # vertical box aligmnent
                              patch_artist=True)   # fill with color
    for patch, color in zip(bplot2['boxes'], colors_6):
        patch.set_facecolor(color)

    print("Colors Set")
    axs[1].yaxis.grid(True)
    axs[1].set_xticks([y+1 for y in range(num_models[3])])
    axs[1].set_xticklabels(labels_6, fontsize = 'x-large')
    axs[1].tick_params(axis='y', which='major', labelsize=12)
    #ax1.set_xlabel('Models', fontsize = 'xx-large')
    axs[1].set_ylabel('sAPE (%)', fontsize = 'xx-large')
    axs[1].set_yscale('log')
    nticks = 30
    maj_loc = matplotlib.ticker.LogLocator(numticks=nticks)
    min_loc = matplotlib.ticker.LogLocator(subs='all', numticks=nticks)
    axs[1].yaxis.set_major_locator(maj_loc)
    axs[1].yaxis.set_minor_locator(min_loc)
    axs[1].set_ylim(bottom = np.min(np.quantile(np.array(smape), 0.05, axis = 1)))
    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)

    # place a text box in upper left in axes coords
    axs[1].text(0.45, 0.1, textstr, transform=axs[1].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)


    for i in range(num_models[3]):
        axs[2].plot(i+1, r2[i], color=colors_6[i], marker='d', linestyle='None', markersize = 12)
    axs[2].hlines(1, 1, num_models[3], linestyles = '--', colors='k')
    #axs[2].hlines(2, 1, num_models, linestyles = '--', colors='r')
    axs[2].yaxis.grid(True)
    axs[2].set_xticks([y+1 for y in range(len(r2))])
    axs[2].set_xticklabels(labels_6, fontsize = 'x-large')
    #ax.set_xlabel('Models', fontsize = 'xx-large')
    axs[2].set_ylabel(r'$R^2$', fontsize = 'xx-large')
    axs[2].set_ylim(top = 1+ 0.1*(1-np.min(r2)))
    axs[2].set_ylim(bottom = np.min(r2) - 0.1*(1-np.min(r2)))
    #textstr = func
    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)

    # place a text box in upper left in axes coords
    axs[2].text(0.05, 0.1, textstr, transform=axs[2].transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    #fig.text(.05, .5, 'Absolute Error', ha='center', va='center', rotation='vertical')
    #plt.ylabel('Absolute Error', fontsize = 'large')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #plt.suptitle("Boxplots of AEs in {} dimensions for four test functions".format(ndims), fontsize = 'xx-large')
    #plt.savefig("plots/3d.png")

    plt.savefig("plots/Mxyzuv.png")
    plt.savefig("plots_pdf/Mxyzuv.pdf", dpi = 150)

    ################################################################################
    ################################################################################
    rows = int(np.ceil(num_models[3]/3))
    cols = 3
    fig, axs = plt.subplots(rows,cols, figsize=(13,6), dpi = 150)  # 1 row, 2 columns
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig.suptitle(r"Prediction versus truth for the $M$ function", fontsize = "xx-large")
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(axs[i, j])

    #axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for i in range(num_models[3]):
        axes[i].scatter(y_test[:5*10**4], y_pred[i][:5*10**4], facecolors='none',
                edgecolors = colors_6[i], s = 50, alpha=0.8, label=labels_6[i])
    #axes[i].scatter(y_test[:5*10**3], y_pred[1][:5*10**3], s = 10, alpha=0.08, label='grid')
    #axes[i].scatter(y_test[:5*10**3], y_pred[2][:5*10**3], s = 10, alpha=0.08, label='rbf')
        #axes[i].set_title("Prediction vs. Truth")
        axes[i].plot(y_test, y_test, 'k--')
        axes[i].set_xscale("symlog", linthreshx = 1e-3)
        axes[i].set_yscale("symlog", linthreshy = 1e-3)
        axes[i].set_xlabel("Truth", fontsize = 'x-large')
        axes[i].set_ylabel("Prediction", fontsize = 'x-large')
        #axes[i].grid(True)
        leg3 = axes[i].legend()
        for lh in leg3.legendHandles:
            lh.set_alpha(1)
    for i in range(rows*cols - num_models[3]):
        axes[i+num_models[3]].set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('truth_vs_pred/truthVSpred_{}.png'.format(NAME_test))
    #plt.savefig('truth_vs_pred/truthVSpred_{}.pdf'.format(NAME_test))



if __name__ == '__main__':
    app.run(main)
