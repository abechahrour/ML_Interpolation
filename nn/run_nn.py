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
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    if drop:
        NAME = NAME + '_dr'
    print(NAME)
    scaler = cm.get_scaler(set_scaler = set_scaler, low = low, high = high, load = load, NAME = NAME)

    ##################################################


    print("Shapes (x, y): ", x_train.shape, y_train.shape)
    print("x_train = ", x_train)
    print("y_train = ", y_train)
    print("x_test = ", x_test)
    print("y_test = ", y_test)
    print("y_train max = ", np.max(y_train))
    print("y_test max = ", np.max(y_test))


    #y = scaler.transform(x_test)

    def lr_scheduler(epoch, lr):
        decay_rate = 0.85
        if epochs < 40:
            return lr
        decay_step = epochs//40
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

    checkpoint_filepath = 'tmp/checkpoint_{}'.format(NAME)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True,
        save_freq=epochs//10)
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
                        restore_best_weights=False,
                        )


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
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        print("y_train after scaling = ", y_train)
        print("y_train max = ", np.max(y_train))
        print("y_train min = ", np.min(y_train))
        print("set_scaler = ", set_scaler)
        print(scaler)
        if activation == 'gelu':
            model = cm.get_model_seq(ndims, opt, nodes = nodes, layers = layers, activation = activation, loss = loss, lr = lr)
            model.summary()
            model = LSUVinit(model, x_train[:batch, :])
        else:
            model = cm.get_model(ndims, opt, nodes = nodes, layers = layers, activation = activation, loss = loss, lr = lr)
            model.summary()



        history = model.fit(x_train, y_train, epochs = epochs, batch_size=batch,
                            verbose = 0, validation_split = 0.1,
                            callbacks=[ckpt, schedule, CustomCallback(), early])
        # history = model.fit(x_train, y_train, epochs = epochs, batch_size=batch,
        #                     verbose = 0, validation_split = 0.1,
        #                     callbacks=[ckpt, lr_schedule, CustomCallback()])
        # tf.keras.backend.set_value(model.optimizer.learning_rate, 0.00001)
        # history2 = model.fit(x_train, y_train, epochs = epochs, batch_size=10**6,
        #                     verbose = 0, validation_split = 0.1,
        #                     callbacks=[ckpt, lr_schedule, CustomCallback()])
        model.load_weights(checkpoint_filepath)
        # opt = SGD(learning_rate=1e-3)
        # model.compile(optimizer = opt,
        #               loss = loss)
        # history3 = model.fit(x_train, y_train, epochs = epochs, batch_size=10**6,
        #                     verbose = 2, validation_split = 0.1,
        #                     callbacks=[ckpt, lr_schedule])
        # model.load_weights(checkpoint_filepath)
        model.save('models/{}'.format(NAME) + '.h5')
        print("Model Saved")
        pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
        with open('history/history_{}'.format(NAME), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print("History Saved")
        #plot_model(model, show_shapes=True, show_layer_names=True)
        history_plot = history.history


    ########### Forward Pass ############################
    start = time.time()
    y_pred = (np.array(model(x_test)).squeeze())
    end = time.time()
    print("Time for prediction = ", end - start)
    y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
    print(y_pred.dtype)

    #print("y_test = ", y_test)
    #print("y_pred = ", y_pred)





    #############################    INTEGRATION     ##########################
    int_naive, std_naive = cm.get_integral_uniform(y_test)
    integral, std = cm.get_integral_uniform(y_pred)

    try:
        int_actual = integral_f(ndims, alpha)
        print("The True Integral = ", int_actual)

        integ = vegas.Integrator(ndims* [[0, 1]])
        int_vegas = integ(f, nitn=10, neval=50000)

        print(int_vegas.summary())
        print('int_vegas = %s    Q = %.2f' % (int_vegas, int_vegas.Q))
        pull = cm.pull(int_actual, integral,std)
    except Exception as e:
        print("Exception tossed: ", e)
        #print("f is not defined")
        int_vegas = 0
        pull = cm.pull(int_naive, integral,std)
    print('Naive Integral = {} +- {}'.format(int_naive, std_naive))
    print('Predicted Integral = {} +- {}'.format(integral, std))
    print("pull = {}".format(pull))

    ############ Compute Measures ###############################
    #logaccuracy
    log_acc = cm.get_logacc(y_test, y_pred)
    relerr = cm.get_relerr(y_test, y_pred)
    mape = np.mean(np.abs(relerr))
    abserr = cm.get_abserr(y_test, y_pred)
    err = cm.get_err(y_test, y_pred)
    mse = cm.get_mse(y_test, y_pred)
    mae = np.mean(abserr)
    sigfigs = cm.get_sigfigs(y_test, y_pred)
    r2 = cm.get_r2(y_test, y_pred)

    pts = []
    eta = []
    for i in range(8):
        pts.append((np.logical_and(sigfigs < -i, sigfigs > -(i+1))).sum())
        eta.append(pts[i]/sigfigs.size*100)

    measures = {
        'mse':mse,
        'mae':mae,
        'err':err,
        'sigfigs':sigfigs,
        'relerr':relerr,
        'mape':mape,
        'func':func,
        'lr':history_plot['lr'],
        'loss':history_plot['loss'],
        'val_loss':history_plot['val_loss'],
        'loss_str':loss,
        'log_acc':log_acc,
        'eta':eta,
        'alpha':alpha
    }

    csv_output = {
        'func':     func,
        'ndims':    [ndims],
        'nodes':    [nodes],
        'layers':   [layers],
        'epochs':   [epochs],
        'batchsize':[batch],
        'activation':activation,
        'loss':     loss,
        'opt':      opt,
        'set_scaler':set_scaler,
        'n_train':  [n_train],
        'n_test':   [n_test],
        'mse':      [mse],
        'mae':      [mae],
        'mape':     [mape],
        'integral':   [integral],
        'std':   [std],
        'int_naive':[int_naive],
        'int_vegas':[int_vegas],
        'pull':     [pull],
        'datetime': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        'r2':       [r2]
    }

    pd.DataFrame.from_dict(data=csv_output).to_csv('nn_runs.csv', mode='a', header = False, index = False)

    cm.plot_all(y_test, y_pred, measures, 'NN', NAME)


if __name__ == '__main__':
    app.run(main)
