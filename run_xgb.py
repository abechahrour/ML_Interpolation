
from xgboost import XGBRegressor
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

from multiprocessing import Process
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
    dir = FLAGS.dir
    func = FLAGS.func
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    set_scaler = ""
    alpha = 0.2
    function = Functions(ndims, alpha)
    if func == 'Gauss':
        f = function.gauss
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
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


    NAME = "{}_n_{}_md_{}_cw_{}_ss_{}_cs_{}_l_{}_d_{}_b_{}_{:.0e}".format(func, n_estimators,
            max_depth, min_child_weight, subsample, colsample_bytree, loss, ndims, booster,n_train)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    print(NAME)

    ##################################################

    print("x_train = ", x_train)
    print("y_train = ", y_train)
    print("x_test = ", x_test)
    print("y_test = ", y_test)
    print("y_train max = ", np.max(y_train))
    print("y_test max = ", np.max(y_test))

    params = {
    # Parameters that we are going to tune.
    'n_estimators':n_estimators,
    'max_depth':max_depth,
    'learning_rate':lr,
    'min_child_weight': min_child_weight,
    'subsample': subsample,
    'colsample_bytree': colsample_bytree,
    'booster':'gbtree',
    'objective':'reg:squarederror',
    'gamma':0

    }
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


    # def fitting():
    #     model = XGBRegressor(**params)
    #     model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
    #              eval_metric='mae',verbose=True, callbacks=[])
    #     # save
    #     filename = 'models/xgbmodel_{}'.format(NAME)
    #     pickle.dump(model, open(file_name, "wb"))
    #     #save the model here on the disk
    #
    # fitting_process = Process(target=fitting)
    # fitting_process.start()
    # fitting_process.join()
    # # load
    # model = pickle.load(open(file_name, "rb"))


    if tf.test.is_gpu_available():
        model = XGBRegressor(**params, tree_method = 'gpu_hist')
    else:
        model = XGBRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
             eval_metric='mae',verbose=True, callbacks=[])
    y_pred = model.predict(x_test)



    # for i in range(100):
    #     #xgb = XGBRegressor()
    #     model = XGBRegressor(**params, random_state=i+2, seed=i+1)
    #     print("about to fit")
    #     model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
    #         eval_metric='mae',verbose=True, callbacks=[])
    #     models.append(model)
    #     y_pred.append(model.predict(x_test))
    # y_pred = np.array(y_pred)
    # print(y_pred.shape)
    # std = np.std(y_pred, axis = 0)
    # print("std = ", std)
    # y_pred = np.mean(y_pred, axis = 0)

    # evaluate the model
    # cv = RepeatedKFold(n_splits=2, n_repeats=2, random_state=1)
    # n_scores = cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
    # print(n_scores)
    #predictions = [round(value) for value in y_pred]


    # evaluate predictions
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: {:.2f}".format(mse))


    #############################    INTEGRATION     ##########################

    integ = vegas.Integrator(ndims* [[0, 1]])
    result = integ(f, nitn=10, neval=50000)

    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    #result_xgb = integ(NN, nitn=10, neval=1000)
    int_naive, err_naive = np.mean(y_test), np.std(y_test)/np.sqrt(y_test.size)
    int_xgb, err_xgb = np.mean(y_pred), np.std(y_pred)/np.sqrt(y_pred.size)
    pull = (int_xgb - result)/err_xgb
    #print(result_xgb.summary())

    print("Integral estimate with {} points".format(n_test))
    print('Naive Integral = {} +- {}'.format(int_naive, err_naive))
    print('XGB Integral = {} +- {}'.format(int_xgb, err_xgb))

    print("pull = {}".format(pull))


    ############ Compute Measures ###############################

    N = 5*10**4

    #logaccuracy

    log_acc = np.log((y_pred/(y_test+1e-9))+1e-9)
    log_acc = log_acc[~np.isnan(log_acc)]
    # MAPE
    mape = ((y_test - y_pred)/(y_test) * 100)
    # mape_interp = ((y_test - y_interp)/(y_test) * 100)
    # mape_rbf = ((y_test - y_rbf)/(y_test) * 100)
    mean_mape = np.mean(np.abs(mape))
    # mean_mape_interp = np.mean(np.abs(mape_interp))
    # mean_mape_rbf = np.mean(np.abs(mape_rbf))

    # Absolute err
    abserr = np.abs(y_test - y_pred)
    # abserr_interp = np.abs(y_test - y_interp)
    # abserr_rbf= np.abs(y_test - y_rbf)

    #Error
    err = (y_test - y_pred)
    # err_interp = (y_test - y_interp)
    # err_rbf = (y_test - y_rbf)

    # Mean Squared Error
    mse = np.mean((y_test-y_pred)**2)
    # mse_interp = np.mean((y_test-y_interp)**2)
    # mse_rbf = np.mean((y_test-y_rbf)**2)

    # Mean Absolute Error
    mae = np.mean(abserr)
    # mae_interp = np.mean(abserr_interp)
    # mae_rbf = np.mean(abserr_rbf)


    #sigfigs = np.log10(np.abs(y_pred/mag(y_pred) - y_test/mag(y_test)))
    sigfigs = np.log10(np.abs(y_pred - y_test))
    # sigfigs_interp = np.log10(np.abs(y_interp/mag(y_interp) - y_test/mag(y_test)))
    # sigfigs_rbf = np.log10(np.abs(y_rbf/mag(y_rbf) - y_test/mag(y_test)))

    pts_01 = (np.logical_and(sigfigs < 0., sigfigs > -1)).sum()
    pts_12 = (np.logical_and(sigfigs < -1, sigfigs > -2)).sum()
    pts_23 = (np.logical_and(sigfigs < -2., sigfigs > -3)).sum()
    pts_34 = (np.logical_and(sigfigs < -3., sigfigs > -4)).sum()
    pts_45 = (np.logical_and(sigfigs < -4., sigfigs > -5)).sum()
    pts_56 = (np.logical_and(sigfigs < -5., sigfigs > -6)).sum()
    pts_67 = (np.logical_and(sigfigs < -6., sigfigs > -7)).sum()
    eta_01 = pts_01/sigfigs.size * 100
    eta_12 = pts_12/sigfigs.size * 100
    eta_23 = pts_23/sigfigs.size * 100
    eta_34 = pts_34/sigfigs.size * 100
    eta_45 = pts_45/sigfigs.size * 100
    eta_56 = pts_56/sigfigs.size * 100
    eta_67 = pts_67/sigfigs.size * 100


    print("y_test sigfigs = ", (y_test/mag(y_test))[:50])
    print("y_pred sigfigs = ", (y_pred/mag(y_pred))[:50])
    # print("y_interp sigfigs = ", (y_interp/mag(y_interp))[:50])
    # print("y_rbf sigfigs = ", (y_rbf/mag(y_rbf))[:50])

    args_y = np.argsort(y_test)

    bins = 100000
    fig, axs = plt.subplots(3,3, figsize=(15,11))  # 1 row, 2 columns
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]
    ax7 = axs[2, 0]
    ax8 = axs[2, 1]
    ax9 = axs[2, 2]

    ax1.hist(sigfigs, histtype='step', label = "XGB", range = (-8, 0))
    # ax1.hist(sigfigs_interp, histtype='step', label = "linear", range = (-15, 0))
    # ax1.hist(sigfigs_rbf, histtype='step', label = "rbf", range = (-11, 0))
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -50
    xhigh = 50
    ax2.set_xlim(-50, 50)

    ax2.hist(mape, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "XGB")
    # ax2.hist(mape_interp, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "linear")
    # ax2.hist(mape_rbf, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "rbf")
    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)

    ax2.text(0.05, 0.90, r'$\epsilon_{{XGB}}$ = {:0.2f} %'.format(mean_mape),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.85, r'MSE$_{{XGB}}$ = {:.1E} '.format(mse),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    if func == 'Gauss':
        ax2.text(0.05, 0.75, r'$\alpha$ = {:.2f} '.format(alpha),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax2.transAxes,
            color='black', fontsize=10)
    # ax2.text(0.05, 0.8, r'MSE$_{{int}}$ = {:.1E} '.format(mse_interp),
    #     verticalalignment='bottom', horizontalalignment='left',
    #     transform=ax2.transAxes,
    #     color='black', fontsize=10)
    # ax2.text(0.05, 0.75, r'MSE$_{{rbf}}$ = {:.1E} '.format(mse_rbf),
    #     verticalalignment='bottom', horizontalalignment='left',
    #     transform=ax2.transAxes,
    #     color='black', fontsize=10)
    ax2.text(0.05, 0.65, r'MAE$_{{XGB}}$ = {:.1E} '.format(mae),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    # ax2.text(0.05, 0.6, r'MAE$_{{int}}$ = {:.1E} '.format(mae_interp),
    #     verticalalignment='bottom', horizontalalignment='left',
    #     transform=ax2.transAxes,
    #     color='black', fontsize=10)
    # ax2.text(0.05, 0.55, r'MAE$_{{rbf}}$ = {:.1E} '.format(mae_rbf),
    #     verticalalignment='bottom', horizontalalignment='left',
    #     transform=ax2.transAxes,
    #     color='black', fontsize=10)

    ax1.text(0.1, 0.90, r'$\eta_{{01}}$ = {:0.1f}%'.format(eta_01),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.text(0.1, 0.85, r'$\eta_{{12}}$ = {:0.1f}%'.format(eta_12),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.text(0.1, 0.8, r'$\eta_{{23}}$ = {:0.1f}%'.format(eta_23),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.text(0.1, 0.75, r'$\eta_{{34}}$ = {:0.1f}%'.format(eta_34),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.text(0.1, 0.7, r'$\eta_{{45}}$ = {:0.1f}%'.format(eta_45),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.text(0.1, 0.65, r'$\eta_{{56}}$ = {:0.1f}%'.format(eta_56),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)
    ax1.text(0.1, 0.60, r'$\eta_{{67}}$ = {:0.1f}%'.format(eta_67),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax1.transAxes,
        color='black', fontsize=10)


    ax3.scatter(y_test[:N], mape[:N], s = 10, alpha=0.5, label = 'XGB')
    # ax3.scatter(y_test[:N], mape_interp[:N], s = 10, alpha=0.08, label = 'interp')
    # ax3.scatter(y_test[:N], mape_rbf[:N], s = 10, alpha=0.08, label = 'rbf')
    ax3.set_title("Difference(%) vs. Truth")
    ax3.hlines(1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.hlines(-1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.set_yscale('symlog', linthreshy = 1e-2)
    ax3.set_xlabel(r"y$_{{true}}$")
    ax3.set_ylabel("Difference(%)")
    ax3.set_ylim(-1e2, 1e2)
    leg3 = ax3.legend()
    for lh in leg3.legendHandles:
        lh.set_alpha(1)
    # log scale for axis Y of the first subplot
    results = model.evals_result()
    epochs = len(results['validation_0']['mae'])
    x_axis = range(0, epochs)
    ax4.set_yscale("log")
    ax4.set_ylabel("Loss ({})".format(loss))
    ax4.set_xlabel("Epochs")
    ax4.set_title("Loss curves")
    #ax0.set_suptitle('Training History')
    #ax4.set_title("Training History \n" + NAME)
    #ax0.plot(x, y, color='r')
    try:
        ax4.plot(x_axis, results['validation_0']['mae'], label='Train')
        ax4.plot(x_axis, results['validation_1']['mae'], label='Test')
    except (KeyError, NameError, UnboundLocalError) as e:
        print('I got a KeyError - reason "%s"' % str(e))

    ax4.grid(True)
    leg4 = ax4.legend(loc=1)

    ax5.scatter(y_test[:N], err[:N], s=10, alpha=0.5, label = 'XGB')
    # ax5.scatter(y_test[:N], err_interp[:N], s=10, alpha=0.08, label='interp')
    # ax5.scatter(y_test[:N], err_rbf[:N], s=10, alpha=0.08, label='rbf')
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy=1e-6)
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    ax6.scatter(y_test[:N], y_pred[:N], s = 10, alpha=0.5, label = 'XGB')
    # ax6.scatter(y_test[:N], y_interp[:N], s = 10, alpha=0.08, label='interp')
    # ax6.scatter(y_test[:N], y_rbf[:N], s = 10, alpha=0.08, label='rbf')
    ax6.set_title("Prediction vs. Truth")
    ax6.plot(y_test, y_test, 'r--')
    #ax6.set_yscale("log")
    ax6.set_xlabel("y$_{{true}}$")
    ax6.set_ylabel("y$_{{pred}}$")
    ax6.grid(True)
    leg6 = ax6.legend()
    for lh in leg6.legendHandles:
        lh.set_alpha(1)

    ax8.scatter(mape[:N], err[:N],s = 10, alpha=0.5, label = 'XGB')
    # ax8.scatter(mape_interp[:N], err_interp[:N],s = 10, alpha=0.08, label='interp')
    # ax8.scatter(mape_rbf[:N], err_rbf[:N],s = 10, alpha=0.08, label='rbf')
    ax8.set_xlabel("Difference(%)")
    ax8.set_ylabel("Error")
    ax8.set_title("Error vs. Difference(%)")
    ax8.set_yscale('symlog', linthreshy=1e-6)
    ax8.set_xscale('symlog', linthreshx=1e-3)
    ax8.set_xlim((-100, 100))
    ax8.grid(True)
    leg8 = ax8.legend()
    for lh in leg8.legendHandles:
         lh.set_alpha(1)
    try:
        ax7.plot(learning_rates, label = "Learning Rate")
    except (KeyError, NameError, UnboundLocalError) as e:
        print('I got a KeyError - reason "%s"' % str(e))
    ax7.set_xlabel("Epochs")
    ax7.set_ylabel("lr")
    ax7.set_yscale('log')
    ax7.set_title("Learning Rate Schedule")


    xlow = -2.5
    xhigh = 2.5
    #ax9.set_xlim(xlow, xhigh)
    # normed_value = 100
    #
    # hist, bins = np.histogram(log_acc, range = (xlow, xhigh), bins=100, density=True)
    # widths = np.diff(bins)
    # hist *= normed_value

    #ax9.bar(bins[:-1], hist, widths, label = "XGB")
    ax9.hist(log_acc, bins = 100, range=(xlow, xhigh), label = "XGB")
    # ax2.hist(mape_interp, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "linear")
    # ax2.hist(mape_rbf, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "rbf")
    ax9.set_yscale('log')
    ax9.set_title("Log Accuracy")
    ax9.set_xlabel(r"$\log\left(\Delta\right)$")
    ax9.set_ylabel("Counts")
    leg9 = ax9.legend()
    for lh in leg9.legendHandles:
        lh.set_alpha(1)


    plt.tight_layout()
    plt.savefig("plots_xgb/{}".format(NAME)+".png")
    plt.savefig("plots_xgb_pdf/{}".format(NAME)+".pdf")



if __name__ == '__main__':
    app.run(main)
