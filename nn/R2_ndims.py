import tensorflow as tf

import sys
sys.path.append('/home/chahrour/Interpolation/')
sys.path.append('/home/chahrour/Interpolation/nn/LSUV-keras')
import pickle

import numpy as np
import time
import itertools
import vegas
import pandas as pd
from Functions import Functions
import common as cm
from datetime import datetime
from lsuv_init import LSUVinit
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_squared_error

from absl import app, flags


tf.keras.backend.set_floatx('float32')
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
    function = Functions(ndims, alpha)



    if load == 0:
        load = False
    elif load == 1:
        load = True


    r2 = []

    for dims in range(2, ndims+1):
        set_scaler = ''
        if dims >= 6:
            set_scaler = 'cube'


        if func == "D3" or func == "D6" or func == "D9" or func == "Mxyzuv":
            _, _, _, x_train, y_train, x_test, y_test = cm.get_functions(func, dims, alpha, n_train, n_test)
        else:
            (f, f_interp, integral_f) = cm.get_functions(func, dims, alpha, n_train, n_test)
            x_train = np.random.rand(n_train, dims)
            x_test = np.random.rand(n_test, dims)
            y_test = f(x_test)
            y_train = f(x_train)
        #ndims = x_train.shape[1]

        NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{:.0e}".format(func, nodes,
                layers, epochs, batch, activation, loss, dims, opt, set_scaler,n_train)
        # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
        #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
        if func == 'Gauss':
            NAME = NAME + '_{:.1f}'.format(alpha)
        if drop:
            NAME = NAME + '_dr'
        print(NAME)

        #scaler = cm.get_scaler(set_scaler = set_scaler, low = low, high = high, load = load, NAME = NAME)

        ##################################################


        #y = scaler.transform(x_test)

        model = tf.keras.models.load_model('models/{}'.format(NAME) + '.h5',
                    compile = False, custom_objects = {activation: cm.gelu})

        scaler = pickle.load(open('scaler/scaler_{}'.format(NAME), 'rb'))



        ########### Forward Pass ############################
        start = time.time()
        y_pred = (np.array(model(x_test)).squeeze())
        end = time.time()
        print("Time for prediction = ", end - start)
        y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
        print(y_pred.dtype)

        #print("y_test = ", y_test)
        #print("y_pred = ", y_pred)





        r2.append(cm.get_r2(y_test, y_pred))

    np.save("r2_vals_{}".format(NAME), r2)

    plt.figure()
    plt.plot(np.arange(2, ndims+1), r2, label = 'rbf')
    plt.xlabel("Dimensions", fontsize = 'x-large')
    plt.ylabel(r"$R^2$", fontsize = 'x-large')
    plt.legend()
    plt.savefig("special_plots/R2_vs_ndims_{}.png".format(NAME))
    plt.savefig("special_plots/R2_vs_ndims_{}.pdf".format(NAME))



if __name__ == '__main__':
    app.run(main)
