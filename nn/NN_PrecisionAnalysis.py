import tensorflow as tf


import pickle

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from absl import app, flags


tf.keras.backend.set_floatx('float32')
FLAGS = flags.FLAGS

flags.DEFINE_string('loss', 'mse', 'The loss function',
                     short_name = 'l')
flags.DEFINE_string('activation', 'elu', 'The Activation',
                     short_name = 'a')
flags.DEFINE_string('dir', 'ti4r4s', 'Directory',
                     short_name = 'dir')
flags.DEFINE_integer('ndims', 2, 'The number of dimensions',
                     short_name='d')
flags.DEFINE_integer('nout', 1, 'The number of outputs',
                     short_name='nout')
flags.DEFINE_integer('epochs', 1000, 'Number of epochs to train',
                     short_name='e')
flags.DEFINE_integer('batch', 100000, 'Number of points to sample per epoch',
                     short_name='b')
flags.DEFINE_integer('nodes', 100, 'Num of nodes',
                     short_name = 'nd')
flags.DEFINE_integer('layers', 8 , 'Exponent of cut function',
                     short_name = 'lyr')
flags.DEFINE_integer('load', 0 , 'Load Model',
                     short_name = 'load')
flags.DEFINE_integer('real', 0 , 'Real or Imag',
                     short_name = 'real')
flags.DEFINE_integer('n_train', 5*10**6 , '# of Training Pts',
                     short_name = 'n_train')
flags.DEFINE_integer('n_test', 5*10**6 , '# of Testing Pts',
                     short_name = 'n_test')




def get_model(nodes = 64, layers = 8, activation = "elu", lr = 1e-3, loss = 'mae'):
    print("loss = ", loss)
    print("activation = ", activation)
    print("nodes = ", nodes)
    print("layers = ", layers)
    print("lr = ", lr)
    initializer = tf.keras.initializers.HeNormal()
    x = Input(shape = (1,))
    h = Dense(nodes, activation=activation, kernel_initializer = initializer)(x)
    #h = BatchNormalization()(h)
    for i in range(layers - 1):
        h = Dense(nodes, activation=activation, kernel_initializer = initializer)(h)
        #h = BatchNormalization()(h)

    y = Dense(1, activation = 'linear')(h)
    model = Model(x, y)
    #model.summary()
    print("The Learning Rate is  ", lr)
    opt = Adam(lr)
    model.compile(optimizer = opt,
                  loss = loss)
    return model

def f(x):
    return np.sin(2*np.pi*x)

def mag(x):
    return 10**(np.floor(np.log10(np.abs(x))))

def main(argv):
    del argv
######################################################
    nodes = FLAGS.nodes
    ndims = FLAGS.ndims
    nout = FLAGS.nout
    layers = FLAGS.layers
    epochs = FLAGS.epochs
    batch = FLAGS.batch
    loss = FLAGS.loss
    load = FLAGS.load
    real = FLAGS.real
    dir = FLAGS.dir
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    activation = FLAGS.activation

    NAME = "Sin_n_{}_l_{}_e_{}_b_{}_{}_{}_{}".format(nodes, layers, epochs, batch, activation, loss, str(n_train))
    print(NAME)
    ##################################################
    x_train = np.array(np.random.uniform(0, 1, n_train), dtype = np.float64)
    x_test = np.array(np.random.uniform(0, 1, n_test), dtype = np.float64)
    y_test = f(x_test)
    y_train = f(x_train)

    def lr_scheduler(epoch, lr):
        decay_rate = 0.2
        decay_step = epochs//10
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=epochs//20, verbose=2,
        mode='auto', min_delta=0.0, cooldown=0, min_lr=1e-9
    )
    model = get_model(nodes, layers, activation = activation, loss = loss)
    history = model.fit(x_train, y_train, epochs = epochs, batch_size=batch, validation_split = 0.2, callbacks=[lr_schedule])
    model.save('models/{}'.format(NAME) + '.h5')
    print("Model Saved")
    with open('history/history_{}'.format(NAME), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("History Saved")
    #plot_model(model, show_shapes=True, show_layer_names=True)

    y_pred = (np.array(model(x_test)).squeeze())
    print(y_pred.dtype)
    relerr = ((y_test - y_pred)/(y_test) * 100)
    mean_err = np.mean(np.abs(relerr))
    sigfigs = np.log10(np.abs(y_pred/mag(y_pred) - y_test/mag(y_test)))
    pts_01 = (np.logical_and(sigfigs < 0., sigfigs > -1)).sum()
    pts_12 = (np.logical_and(sigfigs < -1, sigfigs > -2)).sum()
    pts_23 = (np.logical_and(sigfigs < -2., sigfigs > -3)).sum()
    pts_34 = (np.logical_and(sigfigs < -3., sigfigs > -4)).sum()
    pts_45 = (np.logical_and(sigfigs < -4., sigfigs > -5)).sum()
    pts_56 = (np.logical_and(sigfigs < -5., sigfigs > -6)).sum()
    eta_01 = pts_01/sigfigs.size * 100
    eta_12 = pts_12/sigfigs.size * 100
    eta_23 = pts_23/sigfigs.size * 100
    eta_34 = pts_34/sigfigs.size * 100
    eta_45 = pts_45/sigfigs.size * 100
    eta_56 = pts_56/sigfigs.size * 100


    print((y_pred/mag(y_pred))[:50])
    print((y_test/mag(y_test))[:50])
    args = np.argsort(x_test)

    bins = 100000
    fig, axs = plt.subplots(2,3, figsize=(15,8))  # 1 row, 2 columns
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[0, 2]
    ax4 = axs[1, 0]
    ax5 = axs[1, 1]


    ax1.hist(sigfigs, histtype='step')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim(-5, 5)
    ax2.hist(relerr, histtype = 'step', bins = 100, range = (-5,5))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax2.text(0.35, 0.90, r'$\epsilon$ = {:0.2f} %'.format(mean_err),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes,
        color='green', fontsize=15)
    ax1.text(0.35, 0.90, r'$\eta_{{01}}$ = {:0.1f}%'.format(eta_01),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=15)
    ax1.text(0.35, 0.80, r'$\eta_{{12}}$ = {:0.1f}%'.format(eta_12),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=15)
    ax1.text(0.35, 0.70, r'$\eta_{{23}}$ = {:0.1f}%'.format(eta_23),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=15)
    ax1.text(0.35, 0.60, r'$\eta_{{34}}$ = {:0.1f}%'.format(eta_34),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=15)
    ax1.text(0.35, 0.50, r'$\eta_{{45}}$ = {:0.1f}%'.format(eta_45),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=15)
    ax1.text(0.35, 0.40, r'$\eta_{{56}}$ = {:0.1f}%'.format(eta_56),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=15)

    ax3.plot(x_test[args], y_test[args])
    ax3.plot(x_test[args], y_pred[args])

    # log scale for axis Y of the first subplot
    ax4.set_yscale("log")
    ax4.set_ylabel("Loss")
    ax4.set_xlabel("Epochs")
    #ax0.set_suptitle('Training History')
    #ax4.set_title("Training History \n" + NAME)
    #ax0.plot(x, y, color='r')
    ax4.plot(history.history['loss'], label='Training loss')
    ax4.plot(history.history['val_loss'], label='Training loss')

    ax5.plot(history.history['lr'], label = 'Learning Rate', color='b')
    ax5.set_yscale("log")
    ax5.set_ylabel("lr")
    ax5.set_xlabel("Epochs")
    plt.tight_layout()
    plt.savefig("plots/relerr_{}".format(NAME))


if __name__ == '__main__':
    app.run(main)
