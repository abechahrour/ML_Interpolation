import tensorflow as tf


import pickle

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback, CSVLogger
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from scipy.interpolate import griddata, LinearNDInterpolator
from sklearn.metrics import mean_squared_error
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
from tqdm.keras import TqdmCallback

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
flags.DEFINE_integer('real', 0 , 'Real or Imag',
                     short_name = 'real')
flags.DEFINE_integer('n_train', 5*10**6 , '# of Training Pts',
                     short_name = 'n_train')
flags.DEFINE_integer('n_test', 5*10**6 , '# of Testing Pts',
                     short_name = 'n_test')




def get_model(ndims, opt, nodes, layers, activation, loss, lr = 1e-3):
    print("loss = ", loss)
    print("activation = ", activation)
    print("nodes = ", nodes)
    print("layers = ", layers)
    print("lr = ", lr)

    if activation == 'elu' or activation == 'relu':
        initializer = tf.keras.initializers.HeNormal()
    elif activation == 'tanh' or activation == 'sigmoid':
        initializer = tf.keras.initializers.LecunNormal()
    elif activation == 'selu':
        initializer = tf.keras.initializers.LecunNormal()

    x = Input(shape = (ndims,))
    h1 = Dense(nodes, activation=activation, kernel_initializer = initializer)(x)
    h = Dense(nodes, activation=activation, kernel_initializer = initializer)(h1)
    #h = BatchNormalization()(h)
    for i in range(1, layers - 1):
        # if i%2==0:
        #      h = Add()([h1, h])
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
    #return x
    #return np.mean(x, axis = 1)
    return np.mean(x, axis = 1)*np.prod(np.sin(2*np.pi*x), axis = 1)

def mag(x):
    return 10**(np.floor(np.log10(np.abs(x))))

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
    loss = FLAGS.loss
    load = FLAGS.load
    real = FLAGS.real
    dir = FLAGS.dir
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    activation = FLAGS.activation

    NAME = "Wavey_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}".format(nodes, layers, epochs, batch, activation, loss, ndims, opt, str(n_train))
    print(NAME)
    ##################################################
    x_train = np.random.rand(n_train, ndims)
    num_points = int(n_test**(1/ndims))
    #x_interp = [np.linspace(0, 1, num_points) for i in range(ndims)]
    #grid = np.meshgrid(*x_interp)
    #x_test = np.vstack((*grid)).T
    #y_test = f(x_test)
    x_test = np.random.rand(n_test, ndims)
    y_test = f(x_test)
    y_train = f(x_train)


    y_interp = griddata(points=x_train[:10**3, :], values=y_train[:10**3], xi=x_test, method='linear', fill_value=0)
    y_interp = np.array(y_interp).squeeze()
    print("Interp Done")
    y_rbf = np.array(RBFInterpolator(x_train[:10**3, :], y_train[:10**3])(x_test))
    print("RBF Done")
    #x_interp = np.meshgrid(*[np.linspace(0,1,n_train)[:-1] for i in range(ndims)])
    #print(np.shape(x_interp))
    #print(np.shape(grid))
    #interp = LinearNDInterpolator(np.transpose(list(zip(*x_train))), y_train)
    #y_interp = np.array(interp(*grid)).flatten()
    print(y_interp.shape)
    print(y_rbf.shape)


    def lr_scheduler(epoch, lr):
        decay_rate = 0.2
        decay_step = epochs//10
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.1, patience=10000, verbose=2,
    mode='auto', min_delta=0.00001, cooldown=0, min_lr=0
    )
    checkpoint_filepath = 'tmp/checkpoint_{}'.format(NAME)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True,
        save_freq=epochs//10)

    model = get_model(ndims, opt, nodes = nodes, layers = layers, activation = activation, loss = loss)
    model.summary()
    history = model.fit(x_train, y_train, epochs = epochs, batch_size=batch, verbose = 2, validation_split = 0.0, callbacks=[ckpt, reducelr])
    model.load_weights(checkpoint_filepath)
    model.save('models/{}'.format(NAME) + '.h5')
    print("Model Saved")
    with open('history/history_{}'.format(NAME), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print("History Saved")
    #plot_model(model, show_shapes=True, show_layer_names=True)

    y_pred = (np.array(model(x_test)).squeeze())
    print(y_pred.dtype)



    # MAPE
    mape = ((y_test - y_pred)/(y_test) * 100)
    mape_interp = ((y_test - y_interp)/(y_test) * 100)
    mape_rbf = ((y_test - y_rbf)/(y_test) * 100)
    mean_mape = np.mean(np.abs(mape))
    mean_mape_interp = np.mean(np.abs(mape_interp))
    mean_mape_rbf = np.mean(np.abs(mape_rbf))

    # Absolute err
    abserr = np.abs(y_test - y_pred)
    abserr_interp = np.abs(y_test - y_interp)
    abserr_rbf= np.abs(y_test - y_rbf)

    #Error
    err = (y_test - y_pred)
    err_interp = (y_test - y_interp)
    err_rbf = (y_test - y_rbf)

    # Mean Squared Error
    mse = np.mean((y_test-y_pred)**2)
    mse_interp = np.mean((y_test-y_interp)**2)
    mse_rbf = np.mean((y_test-y_rbf)**2)

    # Mean Absolute Error
    mae = np.mean(abserr)
    mae_interp = np.mean(abserr_interp)
    mae_rbf = np.mean(abserr_rbf)


    sigfigs = np.log10(np.abs(y_pred/mag(y_pred) - y_test/mag(y_test)))
    sigfigs_interp = np.log10(np.abs(y_interp/mag(y_interp) - y_test/mag(y_test)))
    sigfigs_rbf = np.log10(np.abs(y_rbf/mag(y_rbf) - y_test/mag(y_test)))

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


    print("y_test sigfigs = ", (y_test/mag(y_test))[:50])
    print("y_pred sigfigs = ", (y_pred/mag(y_pred))[:50])
    print("y_interp sigfigs = ", (y_interp/mag(y_interp))[:50])
    print("y_rbf sigfigs = ", (y_rbf/mag(y_rbf))[:50])

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

    ax1.hist(sigfigs, histtype='step', label = "NN")
    ax1.hist(sigfigs_interp, histtype='step', label = "interp")
    ax1.hist(sigfigs_rbf, histtype='step', label = "rbf")
    ax1.set_title("Matching Sigfigs")
    ax1.set_yscale('log')
    leg1 = ax1.legend(loc=4)
    for lh in leg1.legendHandles:
        lh.set_alpha(1)

    ax2.set_yscale('log')
    xlow = -50
    xhigh = 50
    ax2.set_xlim(-50, 50)

    ax2.hist(mape, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "NN")
    ax2.hist(mape_interp, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "interp")
    ax2.hist(mape_rbf, histtype = 'step', bins = 100, range = (xlow,xhigh), label = "rbf")
    ax2.set_title("Difference(%)")
    ax2.set_xlabel("Difference(%)")
    leg2 = ax2.legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.08)

    ax2.text(0.05, 0.95, r'$\epsilon_{{NN}}$ = {:0.2f} %'.format(mean_mape),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.85, r'MSE$_{{NN}}$ = {:.1E} '.format(mse),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.8, r'MSE$_{{int}}$ = {:.1E} '.format(mse_interp),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.75, r'MSE$_{{rbf}}$ = {:.1E} '.format(mse_rbf),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.65, r'MAE$_{{NN}}$ = {:.1E} '.format(mae),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.6, r'MAE$_{{int}}$ = {:.1E} '.format(mae_interp),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)
    ax2.text(0.05, 0.55, r'MAE$_{{rbf}}$ = {:.1E} '.format(mae_rbf),
        verticalalignment='bottom', horizontalalignment='left',
        transform=ax2.transAxes,
        color='black', fontsize=10)

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


    ax3.scatter(y_test[:5*10**3], mape[:5*10**3], s = 10, alpha=0.08, label='NN')
    ax3.scatter(y_test[:5*10**3], mape_interp[:5*10**3], s = 10, alpha=0.08, label = 'interp')
    ax3.scatter(y_test[:5*10**3], mape_rbf[:5*10**3], s = 10, alpha=0.08, label = 'rbf')
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
    ax4.set_ylabel("Loss ({})".format(loss))
    ax4.set_xlabel("Epochs")
    ax4.set_title("Loss curves")
    #ax0.set_suptitle('Training History')
    #ax4.set_title("Training History \n" + NAME)
    #ax0.plot(x, y, color='r')
    ax4.plot(history.history['loss'], label='Training loss')
    try:
        ax4.plot(history.history['val_loss'], label='Validation loss')
    except KeyError as e:
        print('I got a KeyError - reason "%s"' % str(e))

    ax4.grid(True)
    leg4 = ax4.legend(loc=1)

    ax5.scatter(y_test[:5*10**3], err[:5*10**3], s=10, alpha=0.08, label='NN')
    ax5.scatter(y_test[:5*10**3], err_interp[:5*10**3], s=10, alpha=0.08, label='interp')
    ax5.scatter(y_test[:5*10**3], err_rbf[:5*10**3], s=10, alpha=0.08, label='rbf')
    ax5.set_title("Error vs. Truth")
    ax5.set_yscale('symlog', linthreshy=1e-6)
    ax5.set_xlabel(r"y$_{{true}}$")
    ax5.set_ylabel("Error")
    ax5.set_ylim(-25, 25)
    ax5.grid(True)
    leg5 = ax5.legend()
    for lh in leg5.legendHandles:
        lh.set_alpha(1)

    ax6.scatter(y_test[:5*10**3], y_pred[:5*10**3], s = 10, alpha=0.08, label='NN')
    ax6.scatter(y_test[:5*10**3], y_interp[:5*10**3], s = 10, alpha=0.08, label='interp')
    ax6.scatter(y_test[:5*10**3], y_rbf[:5*10**3], s = 10, alpha=0.08, label='rbf')
    ax6.set_title("Prediction vs. Truth")
    ax6.plot(y_test, y_test, 'r--')
    #ax6.set_yscale("log")
    ax6.set_xlabel("y$_{{true}}$")
    ax6.set_ylabel("y$_{{pred}}$")
    ax6.grid(True)
    leg6 = ax6.legend()
    for lh in leg6.legendHandles:
        lh.set_alpha(1)

    ax7.scatter(mape[:5*10**3], err[:5*10**3],s = 10, alpha=0.08, label = 'NN')
    ax7.scatter(mape_interp[:5*10**3], err_interp[:5*10**3],s = 10, alpha=0.08, label='interp')
    ax7.scatter(mape_rbf[:5*10**3], err_rbf[:5*10**3],s = 10, alpha=0.08, label='rbf')
    ax7.set_xlabel("Difference(%)")
    ax7.set_ylabel("Error")
    ax7.set_title("Error vs. Difference(%)")
    ax7.set_yscale('symlog', linthreshy=1e-6)
    ax7.set_xscale('symlog', linthreshx=1e-3)
    ax7.set_xlim((-100, 100))
    ax7.grid(True)
    leg3 = ax3.legend()
    for lh in leg3.legendHandles:
         lh.set_alpha(1)
    plt.tight_layout()
    plt.savefig("plots/mape_{}".format(NAME))
    plt.savefig("plots_pdf/mape_{}".format(NAME)+".pdf")


if __name__ == '__main__':
    app.run(main)
