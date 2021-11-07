import numpy as np
import scipy
import pickle
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.interpolate import NearestNDInterpolator
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import pandas as pd
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


def get_box_plot_data(labels, bp, func):
    rows_list = []
    #print("In function", bp['boxes'])
    #print("In function", bp['whiskers'])
    for i in range(len(labels)):
        dict1 = {}
        dict1['label'] = labels[i]
        dict1['lower_whisker'] = bp['whiskers'][i*2].get_ydata()[1]
        dict1['lower_quartile'] = bp['boxes'][i].get_ydata()[1]
        dict1['median'] = bp['medians'][i].get_ydata()[1]
        dict1['upper_quartile'] = bp['boxes'][i].get_ydata()[2]
        dict1['upper_whisker'] = bp['whiskers'][(i*2)+1].get_ydata()[1]
        dict1['mean'] = bp['means'][i].get_ydata()[0]
        dict1['func'] = func
        rows_list.append(dict1)

    return pd.DataFrame(rows_list)

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
    funcs = ['Poly', 'Periodic', 'Camel', 'D{}'.format(ndims)]
    func_names = ['polynomial', 'periodic', 'Camel', 'D{}'.format(ndims)]
    set_scaler = ['', '', 'log', '']
    num_models = [7, 7, 7, 6]
    color = [colors, colors, colors, colors_6]
    method = [methods, methods, methods, methods_6]
    n_test = 1000000
    set_neighbors = 150

    #colors = ['maroon', 'red', 'tomato', 'blue', 'deepskyblue', 'midnightblue']

    y_pred = []
    y_test = []
    for j in range(len(funcs)):
        NAME = "{}_d_{}_{:.0e}_sc_{}".format(funcs[j], ndims, n_train, set_scaler[j])
        NAME_test = NAME +'_{:.0e}_neigh_{}'.format(n_test, set_neighbors)

        #num_models = len(methods)
        if funcs[j] == "D3" or funcs[j] == "D6" or funcs[j] == "D9" or funcs[j] == "Mxyzuv":
            #int_actual, _ = cm.get_integral_uniform(y_test)
            print("Condition satisfied")

            #num_models = len(methods)
        print(NAME_test)
        y_test.append(np.load("predictions/y_test_{}.npy".format(NAME_test)))
        for i in range(num_models[j]):

            y_pred.append(np.load("predictions/y_pred_{}_{}.npy".format(method[j][i+1], NAME_test)))


    print("Predicting values of models\n Computing metrics", flush = True)


    counter = 0
    print("y_pred = ", np.shape(y_pred))
    print("y_test = ", np.shape(y_test))

    for j in range(len(funcs)):

        print(funcs[j])
        for i in range(num_models[j]):
            print(np.shape(y_test[j]))
            print(np.shape(y_pred[counter]))
            #print(y_pred[counter][y_pred[counter] > 1e8])
            #print(y_pred[j][y_pred[counter] > 1e8])
            smape.append(cm.get_smape(y_test[j], y_pred[counter]))
            relerr.append(cm.get_relerr(y_test[j], y_pred[counter]))
            abserr.append(cm.get_abserr(y_test[j], y_pred[counter]))
            r2.append(cm.get_r2(y_test[j], y_pred[counter]))
            counter += 1
            print(counter)

    print("r2 = ", np.shape(r2))
    header_AE = ['label', 'lower_whisker','lower_quartile', 'median', 'upper_quartile',
                'upper_whisker', 'mean', 'func']
    header_sAPE = ['label', 'lower_whisker (%)','lower_quartile (%)', 'median (%)', 'upper_quartile (%)',
                'upper_whisker (%)', 'mean (%)', 'func']
    df_AE = pd.DataFrame(columns = header_AE)
    df_sAPE = pd.DataFrame(columns = header_sAPE)

    df_AE.to_csv('sAPE_{}.csv'.format(ndims), header=header_sAPE)
    df_sAPE.to_csv('AE_{}.csv'.format(ndims), header=header_AE)


    for i in range(4):
        print(funcs[i])
        bplot_sape = plt.boxplot(smape[int(np.sum(num_models[:i])):int(np.sum(num_models[:i]))+num_models[i]], showmeans = True)
        bplot_ae = plt.boxplot(abserr[int(np.sum(num_models[:i])):int(np.sum(num_models[:i]))+num_models[i]], showmeans = True)

        get_box_plot_data(label[i], bplot_sape, funcs[i]).to_csv('sAPE_{}.csv'.format(ndims), mode='a', header=False)
        get_box_plot_data(label[i], bplot_ae, funcs[i]).to_csv('AE_{}.csv'.format(ndims), mode='a', header=False)



    # rows = 2
    # cols = 2
    # fig, axs = plt.subplots(rows, cols, figsize = (14, 8))
    # counter = 0
    # count_func = 0
    # #meanlineprops = dict(linestyle='-', linewidth = 1, color='gold')
    # meanpointprops = dict(marker='D', markeredgecolor='black',
    #                   markerfacecolor='lime')
    # medianlineprops = dict(linestyle='-', linewidth = 1)
    # flierprops = dict(marker='o', markerfacecolor='black', markersize=4,
    #               markeredgecolor='none')
    # for i in range(rows):
    #     for j in range(cols):
    #         for l in range(num_models[count_func]):
    #             y_vals = abserr[int(np.sum(num_models[:count_func])):int(np.sum(num_models[:count_func]))+num_models[count_func]]
    #             print(int(np.sum(num_models[:count_func])+l))
    #             bplot = axs[i, j].boxplot(abserr[int(np.sum(num_models[:count_func]))+l],
    #                                       meanprops = meanpointprops,
    #                                       medianprops=medianlineprops,
    #                                       flierprops = flierprops,
    #                                       meanline=False,
    #                                       showmeans=True,
    #                                       showfliers = False,
    #                                       widths = 0.5,
    #                                       positions = [l+1],
    #                                       notch=False,  # notch shape
    #                                       vert=True,   # vertical box aligmnent
    #                                       patch_artist=True)   # fill with color
    #             #print(color[count_func][l])
    #             #print(bplot)
    #             for patch in bplot['boxes']:
    #                 patch.set_color(color[count_func][l])
    #
    #
    #             counter += 1
    #
    #         xlabels = [label[count_func][i] for i in range(num_models[count_func])]
    #         print("Setting colors")
    #
    #         print("Colors Set")
    #         axs[i, j].yaxis.grid(True)
    #         axs[i, j].set_xticks([y+1 for y in range(num_models[count_func])])
    #         axs[i, j].set_xticklabels(xlabels, fontsize = 'x-large')
    #         axs[i, j].tick_params(axis='y', which='major', labelsize=12)
    #         #ax1.set_xlabel('Models', fontsize = 'xx-large')
    #         axs[i, j].set_ylabel('Absolute Error', fontsize = 'xx-large')
    #         axs[i, j].set_yscale('log')
    #         nticks = 30
    #         maj_loc = matplotlib.ticker.LogLocator(numticks=nticks)
    #         min_loc = matplotlib.ticker.LogLocator(subs='all', numticks=nticks)
    #         axs[i, j].yaxis.set_major_locator(maj_loc)
    #         axs[i, j].yaxis.set_minor_locator(min_loc)
    #         axs[i, j].set_ylim(bottom = np.min(np.quantile(np.array(y_vals), 0.05, axis = 1)))
    #         props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)
    #
    #         # place a text box in upper left in axes coords
    #         axs[i, j].text(0.45, 0.1, funcs[count_func], transform=axs[i, j].transAxes, fontsize=14,
    #             verticalalignment='top', bbox=props)
    #
    #         count_func += 1
    #
    # #fig.text(.05, .5, 'Absolute Error', ha='center', va='center', rotation='vertical')
    # #plt.ylabel('Absolute Error', fontsize = 'large')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.suptitle("Boxplots of AEs in {} dimensions".format(ndims), fontsize = 'xx-large')
    # #plt.savefig("plots/3d.png")
    # plt.savefig("plots/AE_{}d.png".format(ndims))
    # plt.savefig("plots_pdf/AE_{}d.pdf".format(ndims), dpi = 150)
    ###################################################################################

    # rows = 2
    # cols = 2
    # fig, axs = plt.subplots(rows, cols, figsize = (14, 8))
    # counter = 0
    # count_func = 0
    # #meanlineprops = dict(linestyle='-', linewidth = 1, color='slategrey')
    # meanpointprops = dict(marker='D', markeredgecolor='black',
    #                   markerfacecolor='lime')
    # medianlineprops = dict(linestyle='-', linewidth = 1)
    # flierprops = dict(marker='o', markerfacecolor='black', markersize=4,
    #               markeredgecolor='none')
    # for i in range(rows):
    #     for j in range(cols):
    #         for l in range(num_models[count_func]):
    #             print(int(np.sum(num_models[:count_func])+l))
    #             bplot = axs[i, j].boxplot(smape[int(np.sum(num_models[:count_func]))+l],
    #                                      meanprops = meanpointprops,
    #                                      medianprops=medianlineprops,
    #                                      flierprops = flierprops,
    #                                      meanline=False,
    #                                      showmeans=True,
    #                                      showfliers = False,
    #                                      widths = 0.5,
    #                                      positions = [l+1],
    #                                      notch=False,  # notch shape
    #                                      vert=True,   # vertical box aligmnent
    #                                      patch_artist=True)   # fill with color
    #             #print(color[count_func][l])
    #             #print(bplot)
    #             for patch in bplot['boxes']:
    #                 patch.set_color(color[count_func][l])
    #                 #patch.set_color('blue')
    #
    #             print(get_box_plot_data(label[count_func][l], bplot))
    #             counter += 1
    #
    #         xlabels = [label[count_func][i] for i in range(num_models[count_func])]
    #         print("Setting colors")
    #
    #         print("Colors Set")
    #         axs[i, j].yaxis.grid(True)
    #         axs[i, j].set_xticks([y+1 for y in range(num_models[count_func])])
    #         axs[i, j].set_xticklabels(xlabels, fontsize = 'x-large')
    #         axs[i, j].tick_params(axis='y', which='major', labelsize=12)
    #         #ax1.set_xlabel('Models', fontsize = 'xx-large')
    #         axs[i, j].set_ylabel('sAPE (%)', fontsize = 'xx-large')
    #         axs[i, j].set_yscale('log')
    #         axs[i, j].set_ylim(bottom = np.min(np.quantile(np.array(smape[int(np.sum(num_models[:count_func])):int(np.sum(num_models[:count_func]))+num_models[count_func]]), 0.05, axis = 1)))
    #         axs[i, j].set_ylim(top = 300)
    #         props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)
    #
    #         # place a text box in upper left in axes coords
    #         axs[i, j].text(0.45, 0.1, funcs[count_func], transform=axs[i, j].transAxes, fontsize=14,
    #             verticalalignment='top', bbox=props)
    #
    #         # vals = [item.get_ydata() for item in bplot['whiskers']]
    #         # with open("sAPE.dat", 'a+') as f:
    #         #     np.savetxt(f, vals, delimiter = ',')
    #         #np.savetxt('file_2', foo, delimiter=",")
    #
    #         count_func += 1
    # plt.suptitle("Boxplots of sAPEs in {} dimensions".format(ndims), fontsize = 'xx-large')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig("plots/sAPE_{}d.png".format(ndims))
    # plt.savefig("plots_pdf/sAPE_{}d.pdf".format(ndims), dpi = 150)

    ###################################################################################
    ###################################################################################

    # rows = 2
    # cols = 2
    # fig, axs = plt.subplots(rows, cols, figsize = (14, 8))
    # counter = 0
    # count_func = 0
    # meanlineprops = dict(linestyle='-', linewidth = 1, color='slategrey')
    # medianlineprops = dict(linestyle='-', linewidth = 1)
    # flierprops = dict(marker='o', markerfacecolor='black', markersize=4,
    #               markeredgecolor='none')
    # for i in range(rows):
    #     for j in range(cols):
    #         for l in range(num_models[count_func]):
    #             print(int(np.sum(num_models[:count_func])+l))
    #             axs[i, j].plot(l+1, r2[int(np.sum(num_models[:count_func])+l)], color=color[count_func][l], marker='d', linestyle='None', markersize = 12)
    #
    #
    #             counter += 1
    #
    #
    #         start = int(np.sum(num_models[:count_func]))
    #         end = int(np.sum(num_models[:count_func]))+num_models[count_func]
    #         xlabels = [label[count_func][i] for i in range(num_models[count_func])]
    #
    #         axs[i, j].yaxis.grid(True)
    #         axs[i, j].hlines(1, 1, num_models[count_func], linestyles = '--', colors='k')
    #         axs[i, j].set_xticks([y+1 for y in range(num_models[count_func])])
    #         axs[i, j].set_xticklabels(xlabels, fontsize = 'x-large')
    #         axs[i, j].tick_params(axis='y', which='major', labelsize=12)
    #         #ax1.set_xlabel('Models', fontsize = 'xx-large')
    #         axs[i, j].set_ylabel(r'$R^2$', fontsize = 'xx-large')
    #         #axs[i, j].set_yscale('log')
    #         axs[i, j].set_ylim(top = 1 + 0.1*(1-np.min(r2[start:end])))
    #         if np.min(r2[start:end]) < -1:
    #             axs[i, j].set_ylim(bottom = 0)
    #             axs[i, j].set_ylim(top = 1.1)
    #         else:
    #             axs[i, j].set_ylim(bottom = np.min(r2[start:end])- 0.1*(1-np.min(r2[start:end])))
    #         props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)
    #
    #         # place a text box in upper left in axes coords
    #         axs[i, j].text(0.8, 0.1, funcs[count_func], transform=axs[i, j].transAxes, fontsize=14,
    #             verticalalignment='top', bbox=props)
    #         count_func += 1
    # plt.suptitle(r"$R^2$ values in {} dimensions".format(ndims), fontsize = 'xx-large')
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig("plots/R2_{}d.png".format(ndims))
    # plt.savefig("plots_pdf/R2_{}d.pdf".format(ndims), dpi = 150)
    #
    #
    # ################################################################################
    #
    #
    #
    #
    # #axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7]
    # print(np.shape(y_test))
    # print(y_test[0][:5*10**4])
    # print(y_pred[0][:5*10**4])
    # print(np.shape(y_pred))
    # counter = 0
    # counter_2 = 0
    # for num in num_models:
    #     rows = int(np.ceil(num/3))
    #     cols = 3
    #     if num == 6:
    #         fig, axs = plt.subplots(rows,cols, figsize=(13,6), dpi = 150)
    #     else:
    #         fig, axs = plt.subplots(rows,cols, figsize=(13,8), dpi = 150)  # 1 row, 2 columns
    #     #plt.rcParams['font.family'] = 'serif'
    #     #plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    #     fig.suptitle("Prediction versus truth for the {} function".format(func_names[counter_2]), fontsize = "xx-large")
    #     counter_inner = 0
    #     axes = []
    #     for i in range(rows):
    #         for j in range(cols):
    #             axes.append(axs[i, j])
    #     for i in range(num):
    #             axes[i].scatter(y_test[counter_2][:5*10**4], y_pred[counter][:5*10**4], facecolors='none',
    #                     edgecolors = color[counter_2][counter_inner], s = 50, alpha=0.8, label=label[counter_2][counter_inner])
    #
    #             #axes[i].set_title("Prediction vs. Truth")
    #             axes[i].plot(y_test[counter_2], y_test[counter_2], 'k--')
    #             axes[i].set_xscale("symlog", linthreshx = 1e-3)
    #             axes[i].set_yscale("symlog", linthreshy = 1e-3)
    #             axes[i].set_xlabel("Truth", fontsize = 'x-large')
    #             axes[i].set_ylabel("Prediction", fontsize = 'x-large')
    #     #axes[i].grid(True)
    #             leg3 = axes[i].legend()
    #             for lh in leg3.legendHandles:
    #                 lh.set_alpha(1)
    #             counter+=1
    #             counter_inner+=1
    #     for i in range(rows*cols - num):
    #         axes[i+num].set_axis_off()
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     plt.savefig('truth_vs_pred/truthVSpred_{}_{}.png'.format(funcs[counter_2], ndims))
    #     #plt.savefig('truth_vs_pred/truthVSpred_{}_{}.pdf'.format(funcs[counter_2], ndims))
    #     counter_2+=1


if __name__ == '__main__':
    app.run(main)
