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
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
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

    def lr_scheduler(epoch, lr):
        decay_rate = 0.8
        if epochs < 40:
            return lr
        decay_step = 50
        if lr < 1e-8:
            return lr
        elif epoch % decay_step == 0 and epoch:
            print("Reducing lr to ", lr * decay_rate)
            return lr * decay_rate
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.6, patience=100, verbose=2,
    mode='auto', min_delta=1e-9, cooldown=0, min_lr=0
    )
    if opt == 'adam':
        schedule = lr_schedule
    elif opt == 'sgd':
        schedule = reducelr

    class CustomCallback(tf.keras.callbacks.Callback):

        def on_epoch_end(self, epoch, logs=None):
            if epoch%100 == 0:
                keys = list(logs.keys())
                print("End epoch {} of training; loss = {:.4e}, val_loss = {:.4e},lr = {:.4e}".format(epoch,
                        logs["loss"], logs['val_loss'], logs['lr']), flush=True)

    early = EarlyStopping(monitor="val_loss",
                        min_delta=1e-12,
                        patience=epochs//10,
                        verbose=0,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=True,
                        )



    if load == 0:
        load = False
    elif load == 1:
        load = True




    if func == "D3" or func == "D6" or func == "D9" or func == "D9_20" or func == "Mxyzuv":
        _, _, _, x_train, y_train, x_test, y_test = cm.get_functions(func, ndims, alpha, n_train, n_test)
    else:
        (f, f_interp, integral_f) = cm.get_functions(func, ndims, alpha, n_train, n_test)
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
    ndims = x_train.shape[1]

    NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{:.0e}".format(func, nodes,
            layers, epochs, batch, activation, loss, ndims, opt, set_scaler,n_train)
    NAME_func = "{}_d_{}_sc_{}".format(func, ndims, set_scaler)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    if drop:
        NAME = NAME + '_dr'
    print(NAME)
    checkpoint_filepath = 'tmp/checkpoint_{}'.format(NAME)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True,
        save_freq=epochs//40)

    scaler = cm.get_scaler(set_scaler = set_scaler, low = low, high = high, load = 0, NAME = NAME)

    ##################################################


    print("Shapes (x, y): ", x_train.shape, y_train.shape)
    print("x_train = ", x_train)
    print("y_train = ", y_train)
    print("x_test = ", x_test)
    print("y_test = ", y_test)
    print("y_train max = ", np.max(y_train))
    print("y_test max = ", np.max(y_test))
    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    train_sizes = [10**5, 5*10**5, 10**6, 2*10**6, 5*10**6, 10**7, 2*10**7]
    #train_sizes = [10**3, 5*10**3, 10**4, 2*10**4, 5*10**4, 10**5, 2*10**5]

    r2 = []
    mae = []

    for size in train_sizes:
        #batch = size//10**3
        NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{:.0e}".format(func, nodes,
                layers, epochs, batch, activation, loss, ndims, opt, set_scaler,size)

        x = x_train[:size]
        y = y_train[:size]

        print(x.shape)
        print(y.shape)

        print("y_train after scaling = ", y_train)
        print("y_train max = ", np.max(y_train))
        print("y_train min = ", np.min(y_train))
        print("set_scaler = ", set_scaler)
        print(scaler)
        if load:
            if activation == 'gelu':
                model = tf.keras.models.load_model('models/{}'.format(NAME) + '.h5',
                        compile = False, custom_objects = {activation: cm.gelu})
            else:
                model = tf.keras.models.load_model('models/{}'.format(NAME) + '.h5', compile = False)
            history_plot = pickle.load(open('history/history_{}'.format(NAME), "rb"))
            scaler = pickle.load(open('scaler/scaler_{}'.format(NAME), 'rb'))
            print("Model has loaded")
        else:
            if activation == 'gelu':
                model = cm.get_model_seq(ndims, opt, nodes = nodes, layers = layers, activation = activation, loss = loss, lr = lr)
                model.summary()
                model = LSUVinit(model, x[:batch, :])
            else:
                model = cm.get_model(ndims, opt, nodes = nodes, layers = layers, activation = activation, loss = loss, lr = lr)
                model.summary()



            history = model.fit(x, y, epochs = epochs, batch_size=batch,
                                verbose = 0, validation_split = 0.1, shuffle = True,
                                callbacks=[schedule, CustomCallback(), early])

            #model.load_weights(checkpoint_filepath)

            model.save('models/{}'.format(NAME) + '.h5')
            print("Model Saved")
            pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
            with open('history/history_{}'.format(NAME), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
            print("History Saved")
        ########### Forward Pass ############################
        start = time.time()
        y_pred = (np.array(model(x_test)).squeeze())
        end = time.time()
        print("Time for prediction = ", end - start)
        y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
        print(y_pred.dtype)
        r2.append(cm.get_r2(y_test, y_pred))
        mae.append(cm.get_mae(y_test, y_pred))

    r2 = np.array(r2)
    mae = np.array(mae)
    textstr = 'Camel\n9d'
    plt.figure()
    plt.plot(train_sizes, 1 - r2, 'b', label = 'MLP')
    plt.xlabel("# of Training Points", fontsize = 'xx-large')
    plt.ylabel(r"$1 - R^2$", fontsize = 'xx-large')
    plt.legend()
    #plt.title(NAME_func)
    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)

    # place a text box in upper left in axes coords
    plt.text(1.75e7, 7e-3, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("special_plots/R2_vs_ntrain_{}.png".format(NAME_func))
    plt.savefig("special_plots/R2_vs_ntrain_{}.pdf".format(NAME_func))

    plt.figure()
    plt.plot(train_sizes, mae, 'bo', label = 'nn')
    plt.xlabel("# of Training Points", fontsize = 'xx-large')
    plt.ylabel("MAE", fontsize = 'xx-large')
    plt.legend()
    #plt.title(NAME_func)
    props = dict(boxstyle='round', edgecolor='k', facecolor = 'w', linewidth = 2, alpha=1.0)

    # place a text box in upper left in axes coords
    plt.text(1.75e7, 7e-3, textstr, fontsize=14,
        verticalalignment='top', bbox=props)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("special_plots/mae_vs_ntrain_{}.png".format(NAME_func))
    plt.savefig("special_plots/mae_vs_ntrain_{}.pdf".format(NAME_func))



if __name__ == '__main__':
    app.run(main)
