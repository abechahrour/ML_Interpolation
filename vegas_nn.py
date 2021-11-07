import tensorflow as tf


import pickle

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import vegas
import pandas as pd

from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer

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
flags.DEFINE_float('lr', 1e-3, 'The learning rate',
                     short_name = 'lr')
flags.DEFINE_string('func', 'Gauss', 'Function',
                     short_name = 'f')




def get_model(ndims, opt, nodes, layers, activation, loss, lr = 1e-5):
    print("loss = ", loss)
    print("activation = ", activation)
    print("nodes = ", nodes)
    print("layers = ", layers)
    print("lr = ", lr)
    print("opt = ", opt)

    if activation == 'elu' or activation == 'relu':
        initializer = tf.keras.initializers.HeNormal()
    elif activation == 'tanh' or activation == 'sigmoid':
        initializer = tf.keras.initializers.HeNormal()
    elif activation == 'selu':
        initializer = tf.keras.initializers.LecunNormal()


    x = Input(shape = (ndims,))
    h = Dense(nodes, activation=activation, kernel_initializer = initializer)(x)
    #h = BatchNormalization()(h)
    for i in range(layers - 1):
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
        opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif opt == 'rmsprop':
        opt = RMSprop(lr)
    elif opt == 'adagrad':
        opt = Adagrad(lr)
    model.compile(optimizer = opt,
                  loss = loss)
    return model


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
        return np.array(pd.read_csv(path, delimiter=',', nrows = x.shape[0]))

def mag(x):
    return 10**(np.floor(np.log10(np.abs(x))))

def log_transform(x):
    return np.log(x)
def exp_transform(x):
    return np.exp(x)


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
    set_scaler = ""
    alpha = 0.2
    function = Functions(ndims, alpha)
    if func == 'Gauss':
        f = function.gauss
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
        idcs = np.where(y_train > 1e-8)
        x_train = x_train[idcs]
        y_train = y_train[idcs]
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


    NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{:.0e}".format(func, nodes,
            layers, epochs, batch, activation, loss, ndims, opt, set_scaler,n_train)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    print(NAME)

    ##################################################
    low = 0
    high = 1
    if load:
        scaler = pickle.load(open('scaler/scaler_{}'.format(NAME), 'rb'))
    else:
        if set_scaler == "ss":
            scaler = StandardScaler()
        elif set_scaler == "mm":
            scaler = MinMaxScaler((low, high))
        elif set_scaler == "log":
            scaler = FunctionTransformer(log_transform, inverse_func = exp_transform)
        else:
            scaler = FunctionTransformer()



    num_points = int(n_test**(1/ndims))
    #x_interp = [np.linspace(0, 1, num_points) for i in range(ndims)]
    #grid = np.meshgrid(*x_interp)
    #x_test = np.vstack((*grid)).T
    #y_test = f(x_test)
    #print(y_train[:,0] > )

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
        if lr < 1e-7:
            return lr
        elif epoch % decay_step == 0 and epoch:
            print("Reducing lr to ", lr * decay_rate)
            return lr * decay_rate
        return lr

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=0)
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', factor=0.6, patience=epochs//40, verbose=2,
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

    early = EarlyStopping(monitor="loss",
                        min_delta=0,
                        patience=epochs//10,
                        verbose=0,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=False,
                        )


    if load:
        model = tf.keras.models.load_model('models/{}'.format(NAME) + '.h5', compile = False)
        #history_plot = pickle.load(open('history/history_{}'.format(NAME), "rb"))
        #y_train = scaler.transform(y_train.reshape(-1, 1))
        print("Model has loaded")
    else:
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
        model = get_model(ndims, opt, nodes = nodes, layers = layers, activation = activation, loss = loss, lr = lr)
        model.summary()

        history = model.fit(x_train, y_train, epochs = epochs, batch_size=batch,
                            verbose = 1, validation_split = 0.1,
                            callbacks=[early])
        # history = model.fit(x_train, y_train, epochs = epochs, batch_size=batch,
        #                     verbose = 0, validation_split = 0.1,
        #                     callbacks=[ckpt, lr_schedule, CustomCallback()])
        # tf.keras.backend.set_value(model.optimizer.learning_rate, 0.00001)
        # history2 = model.fit(x_train, y_train, epochs = epochs, batch_size=10**6,
        #                     verbose = 0, validation_split = 0.1,
        #                     callbacks=[ckpt, lr_schedule, CustomCallback()])
        #model.load_weights(checkpoint_filepath)
        # opt = SGD(learning_rate=1e-3)
        # model.compile(optimizer = opt,
        #               loss = loss)
        # history3 = model.fit(x_train, y_train, epochs = epochs, batch_size=10**6,
        #                     verbose = 2, validation_split = 0.1,
        #                     callbacks=[ckpt, lr_schedule])
        # model.load_weights(checkpoint_filepath)
        model.save('models/{}'.format(NAME) + '.h5')
        print("Model Saved")
        with open('history/history_{}'.format(NAME), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print("History Saved")
        #plot_model(model, show_shapes=True, show_layer_names=True)
        history_plot = history.history


    y_pred = (np.array(model(x_test)).squeeze())
    y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
    print(y_pred.dtype)

    print("y_test = ", y_test)
    print("y_pred = ", y_pred)




    #############################VEGAS INTEGRATION ##########################

    integ = vegas.Integrator(ndims* [[0, 1]])
    result = integ(f, nitn=10, neval=1000000)

    print(result.summary())
    print('result = %s    Q = %.2f' % (result, result.Q))

    #result_NN = integ(NN, nitn=10, neval=1000)
    int_naive, err_naive = np.mean(y_test), np.std(y_test)/np.sqrt(y_test.size)
    int_nn, err_nn = np.mean(y_pred), np.std(y_pred)/np.sqrt(y_pred.size)
    pull = (int_nn - result)/err_nn
    #print(result_NN.summary())

    print("Integral estimate with {} points".format(n_test))
    print('Naive Integral = {} +- {}'.format(int_naive, err_naive))
    print('NN Integral = {} +- {}'.format(int_nn, err_nn))

    print("pull = {}".format(pull))

if __name__ == '__main__':
    app.run(main)
