import tensorflow as tf


import pickle

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools
import vegas
import keras_tuner as kt

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




def model_builder(hp):
  model = tf.keras.Sequential()


  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  nodes = hp.Int(
      'units',
      min_value=64,
      max_value=128,
      step=32,
      default=128
  )
  activation = hp.Choice(
          'dense_activation',
          values=['tanh', 'elu'],
          default='elu'
      )
  model.add(Dense(units = nodes, input_shape=(3, )))
  for i in range(7):
      model.add(Dense(
            units=nodes,
            activation= activation
        )
        )

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss='mse')

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
    lr = FLAGS.lr
    loss = FLAGS.loss
    load = FLAGS.load
    real = FLAGS.real
    dir = FLAGS.dir
    func = FLAGS.func
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test
    activation = FLAGS.activation
    set_scaler = "mm"
    alpha = 0.5
    function = Functions(ndims, alpha)
    if func == 'Gauss':
        f = function.gauss
    elif func == 'Periodic':
        f = function.periodic
    elif func == 'Poly':
        f = function.poly

    if load == 0:
        load = False
    elif load == 1:
        load = True

    NAME = "hyperparam_search"

    ##################################################
    low = -1
    high = 1
    if set_scaler == "ss":
        scaler = StandardScaler()
    elif set_scaler == "mm":
        scaler = MinMaxScaler((low, high))

    x_train = np.random.rand(n_train, ndims)
    num_points = int(n_test**(1/ndims))
    #x_interp = [np.linspace(0, 1, num_points) for i in range(ndims)]
    #grid = np.meshgrid(*x_interp)
    #x_test = np.vstack((*grid)).T
    #y_test = f(x_test)
    x_test = np.random.rand(n_test, ndims)
    y_test = f(x_test)
    y_train = f(x_train)
    print("x_train = ", x_train)
    print("y_train = ", y_train)
    print("x_test = ", x_test)
    print("y_test = ", y_test)
    print("y_train max = ", np.max(y_train))
    print("y_test max = ", np.max(y_test))

    y_train = scaler.fit_transform(y_train.reshape(-1, 1))
    pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
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

    schedule = reducelr

    checkpoint_filepath = 'tmp/checkpoint_{}'.format(NAME)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='auto',
        save_best_only=True,
        save_freq=epochs//10)

    early = EarlyStopping(monitor="loss",
                        min_delta=0,
                        patience=epochs//10,
                        verbose=0,
                        mode="auto",
                        baseline=None,
                        restore_best_weights=False,
                        )



    # tuner = kt.Hyperband(model_builder,
    #                  objective='val_loss',
    #                  max_epochs=100,
    #                  factor=3,
    #                  directory='my_dir',
    #                  project_name='intro_to_kt')
    tuner = kt.RandomSearch(
            model_builder,
            objective='val_loss',
            max_trials=5)
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs//20)
    tuner.search(x_train, y_train, epochs=epochs, batch_size = batch, validation_split=0.2, callbacks=[stop_early])
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)





if __name__ == '__main__':
    app.run(main)
