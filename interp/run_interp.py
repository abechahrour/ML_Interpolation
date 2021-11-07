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

    def gauss_interp(self, x):

        pre = 1.0/(self.alpha * np.sqrt(np.pi))**self.ndims
        exponent = -1.0*np.sum(((x-0.5)**2)/self.alpha**2, axis=0)
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
    def poly_interp(self, x):
        return np.sum(-x**2 + x, axis=0)

    def periodic_interp(self, x):
        return np.mean(x, axis = 0)*np.prod(np.sin(2*np.pi*x), axis = 0)

    def periodic(self, x):
        return np.mean(x, axis = -1)*np.prod(np.sin(2*np.pi*x), axis = -1)

    def D0(self, x):
        path = '/home/chahrour/Loops/D0000stmmmm/D0000stmmmm_data/D00001rssss_labels_5M.csv'
        return np.array(pd.read_csv(path, delimiter=',', nrows = x.shape[0]))[:,0]




def mag(x):
    return 10**(np.floor(np.log10(np.abs(x))))

def log_transform(x):
    return np.log(x)
def exp_transform(x):
    return np.exp(x)


def main(argv):
    del argv
######################################################

    ndims = FLAGS.ndims

    n_train = FLAGS.n_train
    n_rbf = FLAGS.n_rbf
    n_interp = FLAGS.n_interp
    n_lin = FLAGS.n_lin
    n_grid = FLAGS.n_grid
    n_test = FLAGS.n_test
    func = FLAGS.func
    set_scaler = "ss"
    load = FLAGS.load
    neighbors = 1000

    alpha = 0.2
    function = Functions(ndims, alpha)
    if func == 'Gauss':
        f = function.gauss
        f_interp = function.gauss_interp
        if (alpha < 0.4 and ndims > 3):
            set_scaler = "log"
    elif func == 'Periodic':
        f = function.periodic
        f_interp = function.periodic_interp
    elif func == 'Poly':
        f = function.poly
        f_interp = function.poly_interp
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
    elif func =='D9':
        ndims = 9
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D9_features_6M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D9_labels_6M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        x_train = x[:n_train, :]
        y_train = y[:n_train]*np.prod(x_train, axis = 1)
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]*np.prod(x_test, axis = 1)
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2




    if load == 0:
        load = False
    elif load == 1:
        load = True

    NAME = "{}_d_{}_rbf_{:.0e}_lin_{:.0e}_grid_{:.0e}".format(func, ndims, n_rbf, n_lin, n_grid)
    NAME_rbf = "{}_d_{}_rbf_{:.0e}".format(func, ndims, n_rbf)
    NAME_lin = "{}_d_{}_lin_{:.0e}".format(func, ndims, n_lin)
    NAME_grid = "{}_d_{}_grid_{:.0e}".format(func, ndims, n_grid)
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
        NAME_rbf = NAME_rbf + '_{:.1f}'.format(alpha)
        NAME_lin = NAME_lin + '_{:.1f}'.format(alpha)
        NAME_grid = NAME_grid + '_{:.1f}'.format(alpha)
    print(NAME)
    print("n_train = ", n_train)
    ##################################################

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
        else:
            scaler = FunctionTransformer()
            scaler_grid = FunctionTransformer()


    if func == "D1" or func == "D3" or func == "D6" or func == "D9":
        num_points = int(n_grid**(1/ndims))
        #print("y_grid = ", np.shape(y_grid))
        points = []
        for i in range(ndims):
            points.append(np.linspace(0, 1, num_points))


        y_transf = scaler.fit_transform(y_train.reshape(-1, 1))
        # y_interp = griddata(points=x_train[:n_lin, :], values=y_transf[:n_lin],
        #             xi=x_test, method='linear', fill_value=0)
        tri = Delaunay(x_train[:n_lin, :])  # Compute the triangulation
        #tri_test = Delaunay(x_test)  # Compute the triangulation

        lin_interp = LinearNDInterpolator(tri, y_transf[:n_lin])
        y_interp = np.array(lin_interp(x_test)).squeeze()
        y_interp = np.array(scaler.inverse_transform(y_interp.reshape(-1, 1))).squeeze()
        y_interp_grid = y_interp
        print("Linear Interp done", flush=True)
        #y_interp_grid = np.ones_like(y_rbf)
        #y_interp = np.ones_like(y_rbf)
    else:
        x_train = np.random.rand(n_train, ndims)
        num_points = int(n_grid**(1/ndims))
        #num_points = 2
        print("num_points = ", num_points, flush=True)
        slices = [slice(0, 1, num_points*1j) for i in range(ndims)]
        x_grid = np.mgrid[slices]
        #print(x_grid.shape)
        y_grid = f_interp(x_grid)
        #print("y_grid = ", np.shape(y_grid))
        points = []
        for i in range(ndims):
            points.append(np.linspace(0, 1, num_points))

        #x_interp = [np.linspace(0, 1, num_points) for i in range(ndims)]
        #grid = np.meshgrid(*x_interp)
        #x_test = np.vstack((*grid)).T
        #y_test = f(x_test)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)

        y_transf = scaler.fit_transform(y_train.reshape(-1, 1))
        #y_grid_transf = scaler_grid.fit_transform(y_grid)
        y_interp_grid = interpn(points, y_grid, x_test)
        #y_interp_grid = np.array(scaler_grid.inverse_transform(y_interp_grid)).squeeze()
        print("Grid Interp Done", flush=True)

        if ndims > 3:
            y_interp = y_interp_grid
        else:
            tri = Delaunay(x_train[:n_lin, :])  # Compute the triangulation
            lin_interp = LinearNDInterpolator(tri, y_transf[:n_lin])
            y_interp = np.array(lin_interp(y_test)).squeeze()
            # y_interp = griddata(points=x_train[:n_lin, :], values=y_transf[:n_lin],
            #             xi=x_test, method='linear', fill_value=0)
            # y_interp = np.array(y_interp).squeeze()
            y_interp = np.array(scaler.inverse_transform(y_interp.reshape(-1, 1))).squeeze()
            print("Linear Interp done", flush=True)
            #y_interp_grid = np.ones_like(y_rbf)
            #y_interp = np.ones_like(y_rbf)
        print(y_interp_grid)

    #############  Interpolation ####################################
    print("Starting Interpolation", flush=True)

    ############   RBF Interpolation    ###################
    if n_rbf < 10000:
        rbf_interp = RBFInterpolator(x_train[:n_rbf, :], y_transf[:n_rbf])
        print("RBF interpolator created")
        print("RBF Prediction Starting")
        y_rbf = np.array(rbf_interp(x_test))
    else:
        n_split = 10
        y_rbf = []
        rbf_interp = RBFInterpolator(x_train[:n_rbf, :], y_transf[:n_rbf], neighbors = neighbors)
        print("RBF interpolator created")
        print("RBF Prediction Starting")
        for i in range(n_split):
            y_rbf.append(np.array(rbf_interp(x_test[i*n_test//n_split:(i+1)*n_test//n_split, :])))
        y_rbf = np.ravel(y_rbf)
        NAME = NAME + '_neigh_{:.0e}'.format(neighbors)
        NAME_rbf = NAME_rbf + '_neigh_{:.0e}'.format(neighbors)
        NAME_lin = NAME_lin + '_neigh_{:.0e}'.format(neighbors)
        NAME_grid = NAME_grid + '_neigh_{:.0e}'.format(neighbors)
    print("RBF Done", flush=True)
    y_rbf = np.array(scaler.inverse_transform(y_rbf.reshape(-1, 1))).squeeze()
    print("Interp Done", flush=True)


    y_pred = [y_interp, y_interp_grid, y_rbf]

    pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))

    ################# Save Interpolants #######################

    pickle.dump(rbf_interp, open('interpolants/{}'.format(NAME_rbf), 'wb'))
    pickle.dump(lin_interp, open('scaler/scaler_{}'.format(NAME_lin), 'wb'))
    #pickle.dump(grid_interp, open('interpolants/scaler_{}'.format(NAME_grid), 'wb'))


    #############################    INTEGRATION     ##########################
    int_naive, std_naive = cm.get_integral_uniform(y_test)
    try:
        integ = vegas.Integrator(ndims* [[0, 1]])
        int_vegas = integ(f, nitn=10, neval=50000)

        print(int_vegas.summary())
        print('int_vegas = %s    Q = %.2f' % (int_vegas, int_vegas.Q))
        #pull = cm.pull(int_vegas, int_nn, err_nn)
    except UnboundLocalError:
        print("f is not defined")
        int_vegas = int_naive

    print('Naive Integral = {} +- {}'.format(int_naive, std_naive))

    integrals = []
    stds = []
    pulls = []
    for i in range(np.shape(y_pred)[0]):
        integral, std = cm.get_integral_uniform(y_pred[i])
        pull = cm.pull(int_vegas, integral, std)
        integrals.append(integral)
        stds.append(std)
        pulls.append(pull)
        if i == 2:
            print('RBF Integral = {} +- {}'.format(integrals[i], stds[i]))
            print("pull RBF = {}".format(pulls[i]))
        if i == 1:
            print('Grid Integral (linear) = {} +- {}'.format(integrals[i], stds[i]))
            print("pull Grid = {}".format(pulls[i]))
    #print(result_NN.summary())


    #print(y_rbf)
    #print(y_test)
    #x_interp = np.meshgrid(*[np.linspace(0,1,n_train)[:-1] for i in range(ndims)])
    #print(np.shape(x_interp))
    #print(np.shape(grid))
    #interp = LinearNDInterpolator(np.transpose(list(zip(*x_train))), y_train)
    #y_interp = np.array(interp(*grid)).flatten()
    print(y_interp.shape)
    print(y_rbf.shape)


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
    pts = []
    eta = []
    for i in range(np.shape(y_pred)[0]):
        log_acc.append(cm.get_logacc(y_test, y_pred[i]))
        relerr.append(cm.get_relerr(y_test, y_pred[i]))
        abserr.append(cm.get_abserr(y_test, y_pred[i]))
        mape.append(np.mean(np.abs(relerr[i])))
        err.append(cm.get_err(y_test, y_pred[i]))
        mse.append(cm.get_mse(y_test, y_pred[i]))
        mae.append(np.mean(abserr[i]))
        sigfigs.append(cm.get_sigfigs(y_test, y_pred[i]))
        r2.append(cm.get_r2(y_test, y_pred[i]))

        for j in range(8):
            pts.append((np.logical_and(sigfigs[i] < -j, sigfigs[i] > -(j+1))).sum())
            eta.append(pts[j]/sigfigs[i].size*100)





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
        'n_rbf':  [n_rbf],
        'n_lin':    [n_lin],
        'n_grid':   [n_grid],
        'n_test':   [n_test],
        'mse_rbf':      [mse[2]],
        'mae_rbf':      [mae[2]],
        'mape_rbf':     [mape[2]],
        'mse_lin':      [mse[0]],
        'mae_lin':      [mae[0]],
        'mape_lin':     [mape[0]],
        'mse_grid':      [mse[1]],
        'mae_grid':      [mae[1]],
        'mape_grid':     [mape[1]],
        'int_rbf':   [integrals[2]],
        'err_rbf':   [stds[2]],
        'int_lin':   [integrals[0]],
        'err_lin':   [stds[0]],
        'int_grid':   [integrals[1]],
        'err_grid':   [stds[1]],
        'int_naive':[int_naive],
        'int_vegas':[int_vegas],
        'pull_rbf':     [pulls[2]],
        'pull_interp':     [pulls[0]],
        'pull_grid':     [pulls[1]],
        'datetime': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        'r2_rbf':   [r2[2]],
        'r2_lin':   [r2[0]],
        'r2_grid':   [r2[1]]
    }

    pd.DataFrame.from_dict(data=csv_output).to_csv('interp_runs.csv', mode='a', header = False, index = False)

    #######################################################

    cm.plot_interp(y_test, y_pred, measures, NAME)

if __name__ == '__main__':
    app.run(main)
