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
    num_models = len(methods)
    colors = ['maroon', 'red', 'tomato', 'lightsalmon', 'blue', 'deepskyblue', 'midnightblue']

    NAME = "{}_d_{}_{:.0e}_sc_{}".format(func, ndims, n_train, set_scaler)
    NAME_test = NAME +'_{:.0e}_neigh_{}'.format(n_test, set_neighbors)

    y_test = np.load("predictions/y_test_{}.npy".format(NAME_test))

    if func == "D3" or func == "D6" or func == "D9" or func == "Mxyzuv":
        int_actual, _ = cm.get_integral_uniform(y_test)
        methods = {
            1:'rbf',
            2:'idw',
            3:'nearest',
            4:'lgbm',
            5:'svgp',
            6:'nn'
            }
        num_models = len(methods)
        colors = ['maroon', 'red', 'tomato', 'blue', 'deepskyblue', 'midnightblue']
    else:
        (f, f_interp, integral_f) = cm.get_functions(func, ndims, alpha)
        int_actual = integral_f(ndims, alpha)
    y_pred = []
    for i in range(num_models):
        y_pred.append(np.load("predictions/y_pred_{}_{}.npy".format(methods[i+1], NAME_test)))

    print(np.shape(y_pred[i]))
    print("Predicting values of models\n Computing metrics", flush = True)
    for i in range(num_models):
        print(np.shape(y_pred[i]))
        smape.append(cm.get_smape(y_test, y_pred[i]))
        relerr.append(cm.get_relerr(y_test, y_pred[i]))
        abserr.append(cm.get_abserr(y_test, y_pred[i]))

        integral, std = cm.get_integral_uniform(y_pred[i])
        pull = cm.pull(int_actual, integral, std)
        integrals.append(integral)
        stds.append(std)
        pulls.append(pull)

        r2.append(cm.get_r2(y_test, y_pred[i]))


    print("Predictions and Metrics computed :)", flush = True)
    print("Pulls are ", pulls)
    print("Plotting", flush = True)
    print(np.abs(relerr).shape)
    print(np.shape(relerr))
    xlabels = [methods[i+1] for i in range(num_models)]

    matplotlib.rc('font', family='serif')
    matplotlib.rc('font', serif='Helvetica Neue')
    fig, axs = plt.subplots(2,3, figsize=(16,8), dpi = 150)  # 1 row, 2 columns
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig.suptitle(NAME_test)

    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]

    # rectangular box plot
    # bplot1 = ax1.boxplot(abserr,
    #                          vert=True,   # vertical box aligmnent
    #                          patch_artist=True)   # fill with color

    # notch shape box plot
    bplot1 = ax1.boxplot(abserr,
                             notch=False,  # notch shape
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color
    bplot2 = ax2.boxplot(np.abs(relerr).T,
                             notch=True,  # notch shape
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color
    bplot3 = ax3.boxplot(np.array(smape).T,
                             notch=True,  # notch shape
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color

    # fill with colors

    for bplot in (bplot1, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    # for ax in axes:
    #     #ax.yaxis.grid(True)
    #     ax.set_xticks([y+1 for y in range(len(abserr))], )
    #     ax.set_xlabel('Models')
    #     ax.set_ylabel('Absolute Error')
    # ax1.yaxis.grid(True)
    # ax1.set_xticks([y+1 for y in range(len(abserr))], )
    # ax1.set_xticklabels(xlabels, fontsize = 'x-large')
    # ax1.set_xlabel('Models')
    # ax1.set_ylabel('Absolute Error')

    ax1.yaxis.grid(True)
    ax1.set_xticks([y+1 for y in range(len(abserr))], )
    ax1.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax1.set_xlabel('Models', fontsize = 'xx-large')
    ax1.set_ylabel('Absolute Error', fontsize = 'xx-large')
    ax1.set_yscale('log')
    nticks = 30
    maj_loc = matplotlib.ticker.LogLocator(numticks=nticks)
    min_loc = matplotlib.ticker.LogLocator(subs='all', numticks=nticks)
    ax1.yaxis.set_major_locator(maj_loc)
    ax1.yaxis.set_minor_locator(min_loc)
    #ax1.set_ylim(bottom = 1e-8)
    ax1.set_ylim(bottom = np.min(np.quantile(np.array(abserr), 0.05, axis = 1)))

    ax2.yaxis.grid(True)
    ax2.set_xticks([y+1 for y in range(len(abserr))], )
    ax2.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax2.set_xlabel('Models', fontsize = 'xx-large')
    ax2.set_ylabel('Absolute % Differece', fontsize = 'xx-large')
    ax2.set_yscale('log')


    ax3.yaxis.grid(True)
    ax3.set_xticks([y+1 for y in range(len(smape))], )
    ax3.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax2.set_xlabel('Models', fontsize = 'xx-large')
    ax3.set_ylabel('sMAPE', fontsize = 'xx-large')
    #log_acc = np.concatenate(log_acc, axis = 0)
    #print(log_acc)
    #print(np.shape(log_acc))
    ax3.set_ylim(bottom = np.min(np.quantile(np.array(smape), 0.05, axis = 1)))
    ax3.set_ylim(top = 300)
    ax3.set_yscale('log')


    # for i in range(num_models):
    #     ax3.scatter(y_test[:5*10**3], y_pred[i][:5*10**3], facecolors='none',
    #             edgecolors = colors[i], s = 50, alpha=0.8, label=dict[i+1])
    # #ax3.scatter(y_test[:5*10**3], y_pred[1][:5*10**3], s = 10, alpha=0.08, label='grid')
    # #ax3.scatter(y_test[:5*10**3], y_pred[2][:5*10**3], s = 10, alpha=0.08, label='rbf')
    # ax3.set_title("Prediction vs. Truth")
    # ax3.plot(y_test, y_test, 'k--')
    # ax3.set_xscale("symlog", linthreshx = 1e-3)
    # ax3.set_yscale("symlog", linthreshy = 1e-3)
    # ax3.set_xlabel("y$_{{true}}$")
    # ax3.set_ylabel("y$_{{pred}}$")
    # #ax3.grid(True)
    # leg3 = ax3.legend()
    # for lh in leg3.legendHandles:
    #     lh.set_alpha(1)

    for i in range(num_models):
        ax4.semilogy(i+1, np.abs(pulls[i]), color=colors[i], marker='s', linestyle='None', markersize = 12)
    ax4.hlines(1, 1, num_models, linestyles = '--', colors='k')
    ax4.hlines(2, 1, num_models, linestyles = '--', colors='k')
    ax4.yaxis.grid(True)
    ax4.set_xticks([y+1 for y in range(len(abserr))])
    ax4.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax4.set_xlabel('Models', fontsize = 'xx-large')
    ax4.set_ylabel('Pull', fontsize = 'xx-large')
    ax4.set_ylim(bottom = 0)


    for i in range(num_models):
        ax5.errorbar(i+1, integrals[i], yerr = stds[i], color=colors[i], marker='s', linestyle='None', markersize = 12)
    ax5.hlines(int_actual, 1, num_models, linestyles = '--', colors='k', label = 'True')
    ax5.yaxis.grid(True)
    ax5.set_xticks([y+1 for y in range(len(abserr))])
    ax5.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax5.set_xlabel('Models', fontsize = 'xx-large')
    ax5.set_ylabel('MC Integral', fontsize = 'xx-large')
    #ax3.set_yscale('log')
    # add x-tick labels
    #plt.setp([ax1, ax2, ax3, ax4], xticks=[y+1 for y in range(len(abserr))],
    #         xticklabels=xlabels, fontsize = 'xx-large')
    for i in range(num_models):
        ax6.plot(i+1, r2[i], color=colors[i], marker='s', linestyle='None', markersize = 12)
    ax6.hlines(1, 1, num_models, linestyles = '--', colors='k')
    #ax6.hlines(2, 1, num_models, linestyles = '--', colors='r')
    ax6.yaxis.grid(True)
    ax6.set_xticks([y+1 for y in range(len(abserr))])
    ax6.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax6.set_xlabel('Models', fontsize = 'xx-large')
    ax6.set_ylabel(r'$R^2$', fontsize = 'xx-large')
    ax6.set_ylim(top = 1+ 0.1*(1-np.min(r2)))
    ax6.set_ylim(bottom = np.min(r2) - 0.1*(1-np.min(r2)))




    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/boxplot_{}.png'.format(NAME_test))
    #plt.savefig('plots_pdf/boxplot_{}.pdf'.format(NAME_test))


    # ######################################################################################
    #
    # fig, axs = plt.subplots(1,3, figsize=(12,4))  # 1 row, 2 columns
    # fig.suptitle(NAME_test)
    #
    # ax1 = axs[0]
    # ax2 = axs[1]
    # ax3 = axs[2]
    #
    #
    # # for i in range(num_models):
    # #     ax1.plot(i+1, time_pred[i], color=colors[i], marker='s', linestyle='None', markersize = 12)
    # # #ax1.hlines(1, 0, num_models+1, linestyles = '--', colors='k')
    # # #ax6.hlines(2, 1, num_models, linestyles = '--', colors='r')
    # # ax1.yaxis.grid(True)
    # # ax1.set_xticks([y+1 for y in range(len(abserr))])
    # # ax1.set_xticklabels(xlabels, fontsize = 'large')
    # # #ax1.set_xlabel('Models', fontsize = 'xx-large')
    # # ax1.set_ylabel('Prediction Time (s)', fontsize = 'large')
    # # #ax1.set_ylim(bottom = 0)
    # # ax1.set_yscale("log")
    #
    # for i in range(num_models):
    #     ax2.plot(i+1, ram[i], color=colors[i], marker='s', linestyle='None', markersize = 12)
    # #ax1.hlines(1, 0, num_models+1, linestyles = '--', colors='k')
    # #ax6.hlines(2, 1, num_models, linestyles = '--', colors='r')
    # ax2.yaxis.grid(True)
    # ax2.set_xticks([y+1 for y in range(len(abserr))])
    # ax2.set_xticklabels(xlabels, fontsize = 'large')
    # #ax2.set_xlabel('Models', fontsize = 'xx-large')
    # ax2.set_ylabel('Model Size (MB)', fontsize = 'large')
    # #ax2.set_ylim(bottom = 0)
    # #ax2.set_yscale("log")
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig('performance/performance_{}.png'.format(NAME_test))
    # plt.savefig('plots_pdf/performance_{}.png'.format(NAME_test))


    ################################################################################
    rows = int(np.ceil(num_models/3))
    cols = 3
    fig, axs = plt.subplots(rows,cols, figsize=(16,8), dpi = 150)  # 1 row, 2 columns
    #plt.rcParams['font.family'] = 'serif'
    #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    fig.suptitle(NAME_test)
    axes = []
    for i in range(rows):
        for j in range(cols):
            axes.append(axs[i, j])

    #axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    for i in range(num_models):
        axes[i].scatter(y_test[:5*10**3], y_pred[i][:5*10**3], facecolors='none',
                edgecolors = colors[i], s = 50, alpha=0.8, label=methods[i+1])
    #axes[i].scatter(y_test[:5*10**3], y_pred[1][:5*10**3], s = 10, alpha=0.08, label='grid')
    #axes[i].scatter(y_test[:5*10**3], y_pred[2][:5*10**3], s = 10, alpha=0.08, label='rbf')
        axes[i].set_title("Prediction vs. Truth")
        axes[i].plot(y_test, y_test, 'k--')
        axes[i].set_xscale("symlog", linthreshx = 1e-3)
        axes[i].set_yscale("symlog", linthreshy = 1e-3)
        axes[i].set_xlabel("y$_{{true}}$")
        axes[i].set_ylabel("y$_{{pred}}$")
        #axes[i].grid(True)
        leg3 = axes[i].legend()
        for lh in leg3.legendHandles:
            lh.set_alpha(1)
    for i in range(rows*cols - num_models):
        axes[i+num_models].set_axis_off()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('truth_vs_pred/truthVSpred_{}.png'.format(NAME_test))
    #plt.savefig('plots_pdf/truthVSpred_{}.pdf'.format(NAME_test))


    fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi = 150)
    # notch shape box plot
    bplot = ax.boxplot(np.transpose(abserr),
                             notch=False,  # notch shape
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(abserr))])
    ax.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax.set_xlabel('Models', fontsize = 'xx-large')
    ax.set_ylabel('Absolute Error', fontsize = 'xx-large')
    ax.set_yscale('log')
    nticks = 30
    maj_loc = matplotlib.ticker.LogLocator(numticks=nticks)
    min_loc = matplotlib.ticker.LogLocator(subs='all', numticks=nticks)
    ax.yaxis.set_major_locator(maj_loc)
    ax.yaxis.set_minor_locator(min_loc)
    ax.set_ylim(bottom = np.min(np.quantile(np.array(abserr), 0.05, axis = 1)))
    textstr = func
    props = dict(boxstyle='round', facecolor='turquoise', alpha=1.0)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig('single_plots/abserr_{}.png'.format(NAME_test))
    #plt.savefig('single_plots_pdf/abserr_{}.pdf'.format(NAME_test))

    fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi = 150)
    # notch shape box plot
    bplot = ax.boxplot(np.abs(relerr).T,
                             notch=True,  # notch shape
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(relerr))])
    ax.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax.set_xlabel('Models', fontsize = 'xx-large')
    ax.set_ylabel('Absolute % Differece', fontsize = 'xx-large')
    ax.set_yscale('log')
    textstr = func
    props = dict(boxstyle='round', facecolor='turquoise', alpha=1.0)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig('single_plots/relerr_{}.png'.format(NAME_test))
    #plt.savefig('single_plots_pdf/relerr_{}.pdf'.format(NAME_test))

    fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi = 150)
    # notch shape box plot
    bplot = ax.boxplot(smape,
                             notch=True,  # notch shape
                             vert=True,   # vertical box aligmnent
                             patch_artist=True)   # fill with color
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(smape))])
    ax.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax.set_xlabel('Models', fontsize = 'xx-large')
    ax.set_ylabel('sMAPE', fontsize = 'xx-large')
    ax.set_yscale('log')
    ax.set_ylim(bottom = np.min(np.quantile(np.array(smape), 0.05, axis = 1)))
    ax.set_ylim(top = 300)
    textstr = func
    props = dict(boxstyle='round', facecolor='turquoise', alpha=1.0)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig('single_plots/smape_{}.png'.format(NAME_test))
    #plt.savefig('single_plots_pdf/smape_{}.pdf'.format(NAME_test))

    fig, ax = plt.subplots(1, 1, figsize=(5,4), dpi = 150)
    for i in range(num_models):
        ax.plot(i+1, r2[i], color=colors[i], marker='s', linestyle='None', markersize = 12)
    ax.hlines(1, 1, num_models, linestyles = '--', colors='k')
    #ax.hlines(2, 1, num_models, linestyles = '--', colors='r')
    ax.yaxis.grid(True)
    ax.set_xticks([y+1 for y in range(len(r2))])
    ax.set_xticklabels(xlabels, fontsize = 'x-large')
    #ax.set_xlabel('Models', fontsize = 'xx-large')
    ax.set_ylabel(r'$R^2$', fontsize = 'xx-large')
    ax.set_ylim(top = 1+ 0.1*(1-np.min(r2)))
    ax.set_ylim(bottom = np.min(r2) - 0.1*(1-np.min(r2)))
    textstr = func
    props = dict(boxstyle='round', facecolor='turquoise', alpha=1.0)

    # place a text box in upper left in axes coords
    ax.text(0.05, 0.1, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig('single_plots/r2_{}.png'.format(NAME_test))
    #plt.savefig('single_plots_pdf/r2_{}.pdf'.format(NAME_test))


if __name__ == '__main__':
    app.run(main)
