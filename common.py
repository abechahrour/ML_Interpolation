from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle
import scipy
import pandas as pd
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.util import dispatch
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import sys

sys.path.append('/home/chahrour/Interpolation/')
from Functions import Functions





def get_functions(func, ndims, alpha, n_train = 1000000, n_test = 1000000):
    function = Functions(ndims, alpha)
    if func == 'Gauss':
        f = function.gauss
        f_interp = function.gauss_interp
        integral_f = function.integral_gauss
        if (alpha < 0.4 and ndims > 3):
            set_scaler = "log"
        return f, f_interp, integral_f
    elif func == 'Camel':
        f = function.camel
        f_interp = function.camel_interp
        integral_f = function.integral_camel
        if (alpha < 0.4 and ndims > 3):
            set_scaler = "log"
        return f, f_interp, integral_f
    elif func == 'Periodic':
        f = function.periodic
        f_interp = function.periodic_interp
        integral_f = function.integral_periodic
        return f, f_interp, integral_f
    elif func == 'Poly':
        f = function.poly
        f_interp = function.poly_interp
        integral_f = function.integral_poly
        return f, f_interp, integral_f
    elif func == 'Higgs':
        f = function.higgs_to_lep
        f_interp = function.higgs_to_lep
        integral_f = function.higgs_to_lep
        return f, f_interp, integral_f
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
        x_test = np.array(pd.read_csv(path_feat, delimiter=',', nrows = n_test, skiprows=10**5))
        y_test = np.array(pd.read_csv(path_labels, delimiter=',', nrows = n_test, skiprows=10**5))[:,0]
        y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return f, f_interp, integral_f, x_train, y_train, x_test, y_test
    elif func =='D1':
        ndims = 1
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D1_features_6M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D1_labels_6M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        x_train = x[:n_train, :]
        y_train = y[:n_train]
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return f, f_interp, integral_f, x_train, y_train, x_test, y_test
    elif func =='D3':
        ndims = 3
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D3_features_6M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D3_labels_6M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        x_train = x[:n_train, :]
        y_train = y[:n_train]
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return 1, 1, 1, x_train, y_train, x_test, y_test
    elif func =='D6':
        ndims = 6
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D6_features_6M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D6_labels_6M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        x_train = x[:n_train, :]
        y_train = y[:n_train]
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return 1, 1, 1, x_train, y_train, x_test, y_test
    elif func =='D9':
        ndims = 9
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D9_features_6M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D9_labels_6M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        #args = np.where(np.logical_and.reduce(x>0.1, axis = 1))
        #x = x[args]
        #y = y[args]
        x_train = x[:n_train, :]
        y_train = y[:n_train]
        #*np.prod(x_train, axis = 1)
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2

        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]
        #*np.prod(x_test, axis = 1)
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return 1, 1, 1, x_train, y_train, x_test, y_test
    elif func =='D9_20':
        ndims = 9
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D9_features_20M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D9_labels_20M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        #args = np.where(np.logical_and.reduce(x>0.1, axis = 1))
        #x = x[args]
        #y = y[args]
        x_train = x[:n_train, :]
        y_train = y[:n_train]
        #*np.prod(x_train, axis = 1)
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2

        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]
        #*np.prod(x_test, axis = 1)
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return 1, 1, 1, x_train, y_train, x_test, y_test
    elif func == 'Mxyzuv':
        #ndims = 9
        #f = function.D0
        path_feat = '/home/chahrour/tsil-1.45/Mxyzuv_features_5M.csv'
        path_labels = '/home/chahrour/tsil-1.45/Mxyzuv_labels_5M_test.csv'
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        x = np.array(pd.read_csv(path_feat, delimiter=',', nrows = y.shape[0]))
        x_train = x[:n_train, :]
        y_train = y[:n_train]
        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]
        #*np.prod(x_test, axis = 1)
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
        return 1, 1, 1, x_train, y_train, x_test, y_test


def get_model(ndims, opt, nodes, layers, activation, loss, lr = 1e-5):
    print("loss = ", loss)
    print("activation = ", activation)
    print("nodes = ", nodes)
    print("layers = ", layers)
    print("lr = ", lr)
    print("opt = ", opt)

    initializer = tf.keras.initializers.HeNormal()
    if activation == 'elu':
        initializer = tf.keras.initializers.HeNormal()
        activation = tf.keras.activations.elu
    elif activation == 'relu':
        initializer = tf.keras.initializers.HeNormal()
        activation = tf.keras.activations.relu
    elif activation == 'gelu':
        initializer = tf.keras.initializers.HeNormal()
        activation = gelu
    elif activation == 'swish':
        initializer = tf.keras.initializers.HeNormal()
        activation = tf.keras.activations.swish
    elif activation == 'tanh' or activation == 'sigmoid':
        initializer = tf.keras.initializers.HeNormal()
    elif activation == 'selu':
        initializer = tf.keras.initializers.LecunNormal()

    x = Input(shape = (ndims,))
    h = Dense(nodes, activation=activation, kernel_initializer = initializer)(x)
    #h = BatchNormalization()(h)
    for i in range(layers - 1):
        #h = tf.keras.layers.Dropout(rate)(h)
        h = Dense(nodes, activation=activation, kernel_initializer = initializer)(h)
        #h = BatchNormalization()(h)

    y = Dense(1, activation = 'linear')(h)
    model = Model(x, y)
    #model.summary()

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=lr,
    # decay_steps=1000,
    # decay_rate=0.8)

    print("The Learning Rate is  ", lr)
    if opt == "adam":
        opt = Adam(lr, amsgrad = False)
    elif opt == 'sgd':
        opt = SGD(lr=lr, momentum = 0.9, clipnorm=1.0)
    elif opt == 'rmsprop':
        opt = RMSprop(lr)
    elif opt == 'adagrad':
        opt = Adagrad(lr)
    model.compile(optimizer = opt,
                  loss = loss)
    return model


def get_model_seq(ndims, opt, nodes, layers, activation, loss, lr = 1e-5, drop = False):
    print("loss = ", loss)
    print("activation = ", activation)
    print("nodes = ", nodes)
    print("layers = ", layers)
    print("lr = ", lr)
    print("opt = ", opt)

    initializer = tf.keras.initializers.HeNormal()
    if activation == 'elu':
        initializer = tf.keras.initializers.HeNormal()
        activation = tf.keras.activations.elu
    elif activation == 'relu':
        initializer = tf.keras.initializers.HeNormal()
        activation = tf.keras.activations.relu
    elif activation == 'gelu':
        initializer = tf.keras.initializers.HeNormal()
        activation = gelu
    elif activation == 'tanh' or activation == 'sigmoid':
        initializer = tf.keras.initializers.HeNormal()
    elif activation == 'selu':
        initializer = tf.keras.initializers.LecunNormal()

    #print("The dropout rate = ", rate)
    model = tf.keras.Sequential()
    model.add(Dense(nodes, activation=activation,
                    kernel_initializer = initializer, input_shape=(ndims, )))
    for i in range(layers):
        model.add(Dense(nodes, activation=activation,
                        kernel_initializer = initializer))

    model.add(Dense(1, activation = 'linear'))
    #model.summary()

    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=lr,
    # decay_steps=1000,
    # decay_rate=0.8)

    print("The Learning Rate is  ", lr)
    if opt == "adam":
        opt = Adam(lr, amsgrad = False)
    elif opt == 'sgd':
        opt = SGD(lr=lr, momentum = 0.9, clipnorm=1.0)
    elif opt == 'rmsprop':
        opt = RMSprop(lr)
    elif opt == 'adagrad':
        opt = Adagrad(lr)
    model.compile(optimizer = opt,
                  loss = loss)
    return model


@tf_export("nn.gelu", v1=[])
@dispatch.add_dispatch_support
def gelu(features, approximate=False, name=None):

    with ops.name_scope(name, "Gelu", [features]):
        features = ops.convert_to_tensor(features, name="features")
        if approximate:
          coeff = math_ops.cast(0.044715, features.dtype)
          return 0.5 * features * (
              1.0 + math_ops.tanh(0.7978845608028654 *
                                  (features + coeff * math_ops.pow(features, 3))))
        else:
          return 0.5 * features * (1.0 + math_ops.erf(
              features / math_ops.cast(1.4142135623730951, features.dtype)))




def mag(x):
    return 10**(np.floor(np.log10(np.abs(x))))

def log_transform(x):
    return np.log(x)
def exp_transform(x):
    return np.exp(x)
def cube_root(x):
    return np.sign(x)* np.abs(x)**(1/3)
def cube(x):
    return x**3
def fifth_root(x):
    return np.sign(x)* np.abs(x)**(1/5)
def fifth(x):
    return x**5

def ninth_root(x):
    return np.sign(x)* np.abs(x)**(1/9)
def ninth(x):
    return x**9

def get_scaler(set_scaler = '', low = 0, high = 1, load = 0, NAME = ''):
    if load:
        scaler = pickle.load(open('scaler/scaler_{}'.format(NAME), 'rb'))
    else:
        if set_scaler == "ss":
            scaler = StandardScaler()
            scaler_grid = StandardScaler()
        elif set_scaler == "mm":
            scaler = MinMaxScaler((low, high))
            scaler_grid = MinMaxScaler((low, high))
        elif set_scaler == "log":
            scaler = FunctionTransformer(log_transform, inverse_func = exp_transform)
            scaler_grid = FunctionTransformer(log_transform, inverse_func = exp_transform)
        elif set_scaler == "cube":
            scaler = FunctionTransformer(cube_root, inverse_func = cube)
        elif set_scaler == "fifth":
            scaler = FunctionTransformer(fifth_root, inverse_func = fifth)
        elif set_scaler == "ninth":
            scaler = FunctionTransformer(ninth_root, inverse_func = ninth)
        else:
            scaler = FunctionTransformer()
            scaler_grid = FunctionTransformer()


    return scaler

def get_smape(y_true, y_pred):
    return (2*np.abs(y_true - y_pred)/(np.abs(y_true)+np.abs(y_pred)) * 100)
def get_logacc(y_true, y_pred):
    log_acc = np.log((y_pred/(y_true+1e-9))+1e-9)
    log_acc = log_acc[~np.isnan(log_acc)]
    return np.array(log_acc)
def get_relerr(y_true, y_pred):
    return ((y_true - y_pred)/(y_true) * 100)
def get_abserr(y_true, y_pred):
    return np.abs(y_true - y_pred)
def get_perc_err(y_true, y_pred):
    return np.abs((y_true - y_pred)/y_true)
def get_err(y_true, y_pred):
    return (y_true - y_pred)
def get_mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)
def get_mae(y_true, y_pred):
    return np.mean(np.abs(y_true-y_pred))
def get_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)/y_true))*100
def get_sigfigs(y_true, y_pred):
    return np.log10(np.abs(y_pred - y_true))
def get_min_abserr(y_true, y_pred):
    return np.min(np.abs(y_true - y_pred))
def get_max_abserr(y_true, y_pred):
    return np.max(np.abs(y_true - y_pred))
def get_quantile(y_true, y_pred, q):
    return np.quantile(np.abs(y_true - y_pred), q)
def get_integral_uniform(y):
    return np.mean(y), np.std(y)/np.sqrt(y.size)

def pull(int_true, int_pred, err):
    return (int_true - int_pred)/err

def get_r2(y_true, y_pred):
    #args = np.where(y_pred > 1e8)
    #print(y_pred[args])
    denom = np.sum((y_true - np.mean(y_true))**2)
    num = np.sum((y_true - y_pred)**2)
    print(denom)
    print(num)
    return 1 - (num/denom)

def plot_all(y_test, y_pred, measures, label, NAME):

    N = 10**5
    fig, axs = plt.subplots(3,3, figsize=(17,11))  # 1 row, 2 columns
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]
    ax7 = axs[2, 0]
    ax8 = axs[2, 1]
    ax9 = axs[2, 2]


    fig.suptitle(NAME)

    ax1.hist(measures['sigfigs'], histtype='step', label = label, range = (-8, 0))
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -200
    xhigh = 200
    ax2.set_xlim(xlow, xhigh)

    ax2.hist(measures['relerr'], histtype = 'step', bins = 100, range = (xlow,xhigh), label = label)

    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)

    ax2.text(0.05, 0.90, r'$\epsilon_{{{}}}$ = {:0.2f} %'.format(label, measures['mape']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.85, r'MSE$_{{{}}}$ = {:.1E} '.format(label, measures['mse']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    if measures['func'] == 'Gauss':
        ax2.text(0.05, 0.75, r'$\alpha$ = {:.2f} '.format(measures['alpha']),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax2.transAxes,
            color='black', fontsize=10)
    ax2.text(0.05, 0.65, r'MAE$_{{{}}}$ = {:.1E} '.format(label, measures['mae']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)


    for i in range(np.size(measures['eta'])):
        ax1.text(0.1, 0.9 - i*0.05, r'$\eta_{{{}}}$ = {:0.1f}%'.format(str(i)+str(i+1), measures['eta'][i]),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax1.transAxes,
            color='black', fontsize=10)



    ax3.scatter(y_test[:N], measures['relerr'][:N], s = 10, alpha=0.5, label = label)
    ax3.set_title("Difference(%) vs. Truth")
    ax3.hlines(1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.hlines(-1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.set_yscale('symlog', linthreshy = 1e-2)
    ax3.set_xscale('symlog', linthreshx = 1e-2)
    ax3.set_xlabel(r"y$_{{true}}$")
    ax3.set_ylabel("Difference(%)")
    ax3.set_ylim(-1e2, 1e2)
    leg3 = ax3.legend()
    for lh in leg3.legendHandles:
        lh.set_alpha(1)
    # log scale for axis Y of the first subplot
    ax4.set_yscale("log")
    ax4.set_ylabel("Loss ({})".format(measures['loss_str']))
    ax4.set_xlabel("Epochs")
    ax4.set_title("Loss curves")
    #ax0.set_suptitle('Training History')
    #ax4.set_title("Training History \n" + NAME)
    #ax0.plot(x, y, color='r')
    try:
        ax4.plot(measures['loss'], label='Training loss')
    except (KeyError, NameError, UnboundLocalError) as e:
        print('I got a KeyError - reason "%s"' % str(e))

    try:
        ax4.plot(measures['val_loss'], label='Validation loss')
    except (KeyError, NameError, UnboundLocalError) as e:
        print('I got a KeyError - reason "%s"' % str(e))
    try:
        ax4.plot(history2.history['loss'], label='loss2')
    except NameError as e:
        print('I got a KeyError - reason "%s"' % str(e))

    ax4.grid(True)
    leg4 = ax4.legend(loc=1)

    ax5.scatter(y_test[:N], measures['err'][:N], s=10, alpha=0.5, label = label)
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy=1e-6)
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    ax6.scatter(y_test[:N], y_pred[:N], s = 10, alpha=0.5, label = label)
    ax6.set_title("Prediction vs. Truth")
    ax6.plot(y_test, y_test, 'r--')
    ax6.set_xscale('symlog', linthreshx=1e-4)
    ax6.set_yscale('symlog', linthreshy=1e-4)
    ax6.set_xlabel("y$_{{true}}$")
    ax6.set_ylabel("y$_{{pred}}$")
    ax6.grid(True)
    leg6 = ax6.legend()
    for lh in leg6.legendHandles:
        lh.set_alpha(1)

    ax8.scatter(measures['relerr'][:N], measures['err'][:N],s = 10, alpha=0.5, label = label)
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
        ax7.plot(measures['lr'], label = "Learning Rate")
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

    #ax9.bar(bins[:-1], hist, widths, label = label)
    ax9.hist(measures['log_acc'], bins = 100, range=(xlow, xhigh), label = label)
    ax9.set_yscale('log')
    ax9.set_title("Log Accuracy")
    ax9.set_xlabel(r"$\log\left(\Delta\right)$")
    ax9.set_ylabel("Counts")
    leg9 = ax9.legend()
    for lh in leg9.legendHandles:
        lh.set_alpha(1)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("plots/{}".format(NAME)+".png")
    plt.savefig("plots_pdf/{}".format(NAME)+".pdf")



def plot_svgp(y_test, y_pred, measures, label, NAME):
    N = 10**5
    fig, axs = plt.subplots(3,3, figsize=(17,11))  # 1 row, 2 columns
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]
    ax7 = axs[2, 0]
    ax8 = axs[2, 1]
    ax9 = axs[2, 2]


    fig.suptitle(NAME)

    ax1.hist(measures['sigfigs'], histtype='step', label = label, range = (-8, 0))
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -200
    xhigh = 200
    ax2.set_xlim(xlow, xhigh)

    ax2.hist(measures['relerr'], histtype = 'step', bins = 100, range = (xlow,xhigh), label = label)

    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)

    ax2.text(0.05, 0.90, r'$\epsilon_{{{}}}$ = {:0.2f} %'.format(label, measures['mape']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.85, r'MSE$_{{{}}}$ = {:.1E} '.format(label, measures['mse']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    if measures['func'] == 'Gauss':
        ax2.text(0.05, 0.75, r'$\alpha$ = {:.2f} '.format(measures['alpha']),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax2.transAxes,
            color='black', fontsize=10)
    ax2.text(0.05, 0.65, r'MAE$_{{{}}}$ = {:.1E} '.format(label, measures['mae']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)


    for i in range(np.size(measures['eta'])):
        ax1.text(0.1, 0.9 - i*0.05, r'$\eta_{{{}}}$ = {:0.1f}%'.format(str(i)+str(i+1), measures['eta'][i]),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax1.transAxes,
            color='black', fontsize=10)



    ax3.scatter(y_test[:N], measures['relerr'][:N], s = 10, alpha=0.5, label = label)
    ax3.set_title("Difference(%) vs. Truth")
    ax3.hlines(1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.hlines(-1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.set_yscale('symlog', linthreshy = 1e-2)
    ax3.set_xscale('symlog', linthreshx = 1e-2)
    ax3.set_xlabel(r"y$_{{true}}$")
    ax3.set_ylabel("Difference(%)")
    ax3.set_ylim(-1e2, 1e2)
    leg3 = ax3.legend()
    for lh in leg3.legendHandles:
        lh.set_alpha(1)
    # log scale for axis Y of the first subplot
    ax4.plot(measures['logf'])
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("ELBO")
    ax4.set_yscale('symlog')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    ax5.scatter(y_test[:N], measures['err'][:N], s=10, alpha=0.5, label = label)
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy=1e-6)
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    ax6.scatter(y_test[:N], y_pred[:N], s = 10, alpha=0.5, label = label)
    ax6.set_title("Prediction vs. Truth")
    ax6.plot(y_test, y_test, 'r--')
    ax6.set_xscale('symlog', linthreshx=1e-4)
    ax6.set_yscale('symlog', linthreshy=1e-4)
    ax6.set_xlabel("y$_{{true}}$")
    ax6.set_ylabel("y$_{{pred}}$")
    ax6.grid(True)
    leg6 = ax6.legend()
    for lh in leg6.legendHandles:
        lh.set_alpha(1)


    xlow = -2.5
    xhigh = 2.5
    ax7.hist(measures['log_acc'], bins = 100, range=(xlow, xhigh), label = label)
    ax7.set_yscale('log')
    ax7.set_title("Log Accuracy")
    ax7.set_xlabel(r"$\log\left(\Delta\right)$")
    ax7.set_ylabel("Counts")
    leg7 = ax7.legend()
    for lh in leg7.legendHandles:
        lh.set_alpha(1)

    ax8.scatter(measures['relerr'][:N], measures['err'][:N],s = 10, alpha=0.5, label = label)
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

    ax9.plot(measures['mae_tr'], label = 'Training')
    ax9.plot(measures['mae_val'], label = 'Validation')
    ax9.set_xlabel("Epoch")
    ax9.set_ylabel("mae")
    ax9.set_yscale("log")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.legend()

    #ax9.set_xlim(xlow, xhigh)
    # normed_value = 100
    #
    # hist, bins = np.histogram(log_acc, range = (xlow, xhigh), bins=100, density=True)
    # widths = np.diff(bins)
    # hist *= normed_value

    #ax9.bar(bins[:-1], hist, widths, label = label)



    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("plots_svgp/{}".format(NAME)+".png")
    plt.savefig("plots_svgp_pdf/{}".format(NAME)+".pdf")
def plot_nearest(y_test, y_pred, measures, label, NAME):
    N = 10**5
    fig, axs = plt.subplots(2,3, figsize=(15,8))  # 1 row, 2 columns
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]
    # ax7 = axs[2, 0]
    # ax4 = axs[2, 1]
    # ax9 = axs[2, 2]
    fig.suptitle(NAME)

    #ax1.hist(measures['sigfigs'], histtype='step', label = "NN", range = (-15, 0))
    ax1.hist(measures['sigfigs'], histtype='step', label = label, range = (-9, 0))
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -200
    xhigh = 200
    ax2.set_xlim(xlow, xhigh)

    #ax2.hist(mape, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "NN")
    ax2.hist(measures['relerr'], histtype = 'step', bins = 100, range = (xlow,xhigh), label = label)
    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)


    ax2.text(0.05, 0.80, r'$\epsilon$ = {:0.2f} %'.format(measures['mape']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)


    ax2.text(0.05, 0.6, r'MSE = {:.1E} '.format(measures['mse']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)


    ax2.text(0.05, 0.4, r'MAE = {:.1E} '.format(measures['mae']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)




    #ax3.scatter(y_test[:5*10**3], mape[:5*10**3], s = 10, alpha=0.08, label='NN')
    ax3.scatter(y_test[:5*10**3], measures['relerr'][:5*10**3], s = 10, alpha=0.08, label = label)
    ax3.set_title("Difference(%) vs. Truth")
    ax3.hlines(1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.hlines(-1, np.min(y_test), np.max(y_test), colors = 'red', linestyles = 'dashed')
    ax3.set_yscale('symlog', linthreshy = 1e-2)
    if np.max(y_test[:5*10**3]) > 1:
        ax3.set_xscale('symlog', linthreshx = 1e-2)
    else:
        ax3.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax3.set_xlabel(r"y$_{{true}}$")
    ax3.set_ylabel("Difference(%)")
    ax3.set_ylim(-1e2, 1e2)

    leg3 = ax3.legend()
    for lh in leg3.legendHandles:
        lh.set_alpha(1)



    #ax5.scatter(y_test[:5*10**3], err[:5*10**3], s=10, alpha=0.08, label='NN')
    ax5.scatter(y_test[:5*10**3], measures['err'][:5*10**3], s=10, alpha=0.08, label=label)
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy = 1e-5)
    if np.max(y_test[:5*10**3]) > 1:
        ax5.set_xscale('symlog', linthreshx = 1e-5)
    else:
        ax5.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    #ax6.scatter(y_test[:5*10**3], y_pred[:5*10**3], s = 10, alpha=0.08, label='NN')
    ax6.scatter(y_test[:5*10**3], y_pred[:5*10**3], s = 10, alpha=0.08, label=label)
    ax6.set_title("Prediction vs. Truth")
    ax6.plot(y_test, y_test, 'r--')
    if np.max(y_test[:5*10**3]) > 1:
        ax6.set_yscale("symlog", linthreshy = 1e-2)
        ax6.set_xscale("symlog", linthreshx = 1e-2)
    else:
        ax6.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax6.set_xlabel("y$_{{true}}$")
    ax6.set_ylabel("y$_{{pred}}$")
    ax6.grid(True)
    leg6 = ax6.legend()
    for lh in leg6.legendHandles:
        lh.set_alpha(1)

    #ax4.scatter(mape[:5*10**3], err[:5*10**3],s = 10, alpha=0.08, label = 'NN')
    ax4.scatter(measures['relerr'][:5*10**3], measures['err'][:5*10**3],s = 10, alpha=0.08, label=label)
    ax4.set_xlabel("Difference(%)")
    ax4.set_ylabel("Error")
    ax4.set_title("Error vs. Difference(%)")
    ax4.set_yscale('symlog', linthreshy=1e-6)
    ax4.set_xscale('symlog', linthreshx=1e-3)
    ax4.set_xlim((-100, 100))
    ax4.grid(True)
    leg8 = ax4.legend()
    for lh in leg8.legendHandles:
         lh.set_alpha(1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots/{}".format(NAME)+".png")
    plt.savefig("plots_pdf/{}".format(NAME)+".pdf")

def plot_interp(y_test, y_pred, measures, NAME):
    N = 10**5
    fig, axs = plt.subplots(2,3, figsize=(15,8))  # 1 row, 2 columns
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]
    ax6 = axs[1, 2]
    # ax7 = axs[2, 0]
    # ax4 = axs[2, 1]
    # ax9 = axs[2, 2]
    fig.suptitle(NAME)

    #ax1.hist(measures['sigfigs'], histtype='step', label = "NN", range = (-15, 0))
    ax1.hist(measures['sigfigs'][0], histtype='step', label = "linear", range = (-9, 0))
    ax1.hist(measures['sigfigs'][1], histtype='step', label = "grid", range = (-9, 0))
    ax1.hist(measures['sigfigs'][2], histtype='step', label = "rbf", range = (-9, 0))
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -200
    xhigh = 200
    ax2.set_xlim(xlow, xhigh)

    #ax2.hist(mape, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "NN")
    ax2.hist(measures['relerr'][0], histtype = 'step', bins = 100, range = (xlow,xhigh), label = "linear")
    ax2.hist(measures['relerr'][1], histtype = 'step', bins = 100, range = (xlow,xhigh), label = "grid")
    ax2.hist(measures['relerr'][2], histtype = 'step', bins = 100, range = (xlow,xhigh), label = "rbf")
    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)


    ax2.text(0.05, 0.80, r'$\epsilon_{{{}}}$ = {:0.2f} %'.format('lin', measures['mape'][0]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.75, r'$\epsilon_{{{}}}$ = {:0.2f} %'.format('grid', measures['mape'][1]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.70, r'$\epsilon_{{{}}}$ = {:0.2f} %'.format('rbf', measures['mape'][2]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)

    ax2.text(0.05, 0.6, r'MSE$_{{lin}}$ = {:.1E} '.format(measures['mse'][0]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.55, r'MSE$_{{grid}}$ = {:.1E} '.format(measures['mse'][1]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.50, r'MSE$_{{rbf}}$ = {:.1E} '.format(measures['mse'][2]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)

    ax2.text(0.05, 0.4, r'MAE$_{{lin}}$ = {:.1E} '.format(measures['mae'][0]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.35, r'MAE$_{{grid}}$ = {:.1E} '.format(measures['mae'][1]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.30, r'MAE$_{{rbf}}$ = {:.1E} '.format(measures['mae'][2]),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)



    #ax3.scatter(y_test[:5*10**3], mape[:5*10**3], s = 10, alpha=0.08, label='NN')
    ax3.scatter(y_test[:5*10**3], measures['relerr'][0][:5*10**3], s = 10, alpha=0.08, label = 'interp')
    ax3.scatter(y_test[:5*10**3], measures['relerr'][1][:5*10**3], s = 10, alpha=0.08, label = 'grid')
    ax3.scatter(y_test[:5*10**3], measures['relerr'][2][:5*10**3], s = 10, alpha=0.08, label = 'rbf')
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



    #ax5.scatter(y_test[:5*10**3], err[:5*10**3], s=10, alpha=0.08, label='NN')
    ax5.scatter(y_test[:5*10**3], measures['err'][0][:5*10**3], s=10, alpha=0.08, label='interp')
    ax5.scatter(y_test[:5*10**3], measures['err'][1][:5*10**3], s=10, alpha=0.08, label='grid')
    ax5.scatter(y_test[:5*10**3], measures['err'][2][:5*10**3], s=10, alpha=0.08, label='rbf')
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy=1e-6)
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    #ax6.scatter(y_test[:5*10**3], y_pred[:5*10**3], s = 10, alpha=0.08, label='NN')
    ax6.scatter(y_test[:5*10**3], y_pred[0][:5*10**3], s = 10, alpha=0.08, label='interp')
    ax6.scatter(y_test[:5*10**3], y_pred[1][:5*10**3], s = 10, alpha=0.08, label='grid')
    ax6.scatter(y_test[:5*10**3], y_pred[2][:5*10**3], s = 10, alpha=0.08, label='rbf')
    ax6.set_title("Prediction vs. Truth")
    ax6.plot(y_test, y_test, 'r--')
    #ax6.set_yscale("log")
    ax6.set_xlabel("y$_{{true}}$")
    ax6.set_ylabel("y$_{{pred}}$")
    ax6.grid(True)
    leg6 = ax6.legend()
    for lh in leg6.legendHandles:
        lh.set_alpha(1)

    #ax4.scatter(mape[:5*10**3], err[:5*10**3],s = 10, alpha=0.08, label = 'NN')
    ax4.scatter(measures['relerr'][0][:5*10**3], measures['err'][0][:5*10**3],s = 10, alpha=0.08, label='interp')
    ax4.scatter(measures['relerr'][1][:5*10**3], measures['err'][1][:5*10**3],s = 10, alpha=0.08, label='grid')
    ax4.scatter(measures['relerr'][2][:5*10**3], measures['err'][2][:5*10**3],s = 10, alpha=0.08, label='rbf')
    ax4.set_xlabel("Difference(%)")
    ax4.set_ylabel("Error")
    ax4.set_title("Error vs. Difference(%)")
    ax4.set_yscale('symlog', linthreshy=1e-6)
    ax4.set_xscale('symlog', linthreshx=1e-3)
    ax4.set_xlim((-100, 100))
    ax4.grid(True)
    leg8 = ax4.legend()
    for lh in leg8.legendHandles:
         lh.set_alpha(1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots_interp/{}".format(NAME)+".png")
    plt.savefig("plots_pdf_interp/{}".format(NAME)+".pdf")

def plot_lgbm(y_test, y_pred, measures, label, NAME):
    N = 10**5

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

    fig.suptitle(NAME)

    ax1.hist(measures['sigfigs'], histtype='step', label = label, range = (-8, 0))
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -200
    xhigh = 200
    ax2.set_xlim(xlow, xhigh)

    ax2.hist(measures['relerr'], histtype = 'step', bins = 100, range = (xlow,xhigh), label = label)
    # ax2.hist(mape_interp, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "linear")
    # ax2.hist(mape_rbf, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "rbf")
    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)

    ax2.text(0.05, 0.90, r'$\epsilon_{{{}}}$ = {:0.2f} %'.format(label, measures['mape']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.85, r'MSE$_{{{}}}$ = {:.1E} '.format(label, measures['mse']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    if measures['func'] == 'Gauss':
        ax2.text(0.05, 0.75, r'$\alpha$ = {:.2f} '.format(measures['alpha']),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax2.transAxes,
            color='black', fontsize=10)
    ax2.text(0.05, 0.65, r'MAE$_{{{}}}$ = {:.1E} '.format(label, measures['mae']),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)

    for i in range(np.size(measures['eta'])):
        ax1.text(0.1, 0.9 - i*0.05, r'$\eta_{{{}}}$ = {:0.1f}%'.format(str(i)+str(i+1), measures['eta'][i]),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax1.transAxes,
            color='black', fontsize=10)

    ax3.scatter(y_test[:N], measures['relerr'][:N], s = 10, alpha=0.5, label = label)
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
    ax4.set_yscale("log")
    ax4.set_ylabel("Loss ({})".format(measures['loss']))
    ax4.set_xlabel("Epochs")
    ax4.set_title("Loss curves")
    #ax0.set_suptitle('Training History')
    #ax4.set_title("Training History \n" + NAME)
    #ax0.plot(x, y, color='r')
    try:
        lgb.plot_metric(measures['model'], metric = measures['loss'], ax = ax4)
        #lgb.plot_metric(measures['model'], metric = 'l1', ax = ax4)
    except (KeyError, NameError, UnboundLocalError) as e:
        print('I got a KeyError - reason "%s"' % str(e))

    ax4.grid(True)
    leg4 = ax4.legend(loc=1)

    ax5.scatter(y_test[:N], measures['err'][:N], s=10, alpha=0.5, label = label)
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy=1e-6)
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    ax6.scatter(y_test[:N], y_pred[:N], s = 10, alpha=0.5, label = label)
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

    ax8.scatter(measures['relerr'][:N], measures['err'][:N],s = 10, alpha=0.5, label = label)
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

    xlow = -2.5
    xhigh = 2.5
    #ax9.set_xlim(xlow, xhigh)
    # normed_value = 100
    #
    # hist, bins = np.histogram(log_acc, range = (xlow, xhigh), bins=100, density=True)
    # widths = np.diff(bins)
    # hist *= normed_value

    #ax9.bar(bins[:-1], hist, widths, label = "LGBM")
    ax9.hist(measures['log_acc'], bins = 100, range=(xlow, xhigh), label = label)
    # ax2.hist(mape_interp, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "linear")
    # ax2.hist(mape_rbf, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "rbf")
    ax9.set_yscale('log')
    ax9.set_title("Log Accuracy")
    ax9.set_xlabel(r"$\log\left(\Delta\right)$")
    ax9.set_ylabel("Counts")
    leg9 = ax9.legend()
    for lh in leg9.legendHandles:
        lh.set_alpha(1)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("plots_lgbm/{}".format(NAME)+".png")
    plt.savefig("plots_lgbm_pdf/{}".format(NAME)+".pdf")
