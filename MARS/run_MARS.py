import tensorflow as tf

import sys
sys.path.append('/home/chahrour/Interpolation/')
import pickle

import numpy as np
import time
import itertools
import vegas
import pandas as pd
from Functions import Functions
import common as cm
from datetime import datetime
from pyearth import Earth

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
        opt = SGD(lr=lr, momentum = 0.9, clipnorm=1.0)
    elif opt == 'rmsprop':
        opt = RMSprop(lr)
    elif opt == 'adagrad':
        opt = Adagrad(lr)
    model.compile(optimizer = opt,
                  loss = loss)
    return model





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
    low = 0
    high = 1
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
        # if ndims > 5:
        #     set_scaler = ""
        #     low = -100
        #     high = 100

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
        #y_train = y[:n_train]
        y_train = y[:n_train]*np.prod(x_train, axis = 1)
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = x[n_train:n_train+n_test, :]
        #y_test = y[n_train:n_train+n_test]
        y_test = y[n_train:n_train+n_test]*np.prod(x_test, axis = 1)
        #y_test = y_test * x_test[:,0] * x_test[:,1]**2
    elif func =='D6':
        ndims = 6
        #f = function.D0
        path_feat = '/home/chahrour/Interpolation/D0_data/D6_features_6M.csv'
        path_labels = '/home/chahrour/Interpolation/D0_data/D6_labels_6M.csv'
        x = np.array(pd.read_csv(path_feat, delimiter=','))
        y = np.array(pd.read_csv(path_labels, delimiter=','))[:,0]
        x_train = x[:n_train, :]
        y_train = y[:n_train]*np.prod(x_train, axis = 1)
        #y_train = y_train * x_train[:,0] * x_train[:,1]**2
        x_test = x[n_train:n_train+n_test, :]
        y_test = y[n_train:n_train+n_test]*np.prod(x_test, axis = 1)
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


    NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{:.0e}".format(func, nodes,
            layers, epochs, batch, activation, loss, ndims, opt, set_scaler,n_train)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
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


    if load:
        model = tf.keras.models.load_model('models/{}'.format(NAME) + '.h5', compile = False)
        history_plot = pickle.load(open('history/history_{}'.format(NAME), "rb"))
        #scaler = pickle.load(open('scaler/scaler_{}'.format(NAME), 'rb'))
        print("Model has loaded")
    else:
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        print("y_train after scaling = ", y_train)
        print("y_train max = ", np.max(y_train))
        print("y_train min = ", np.min(y_train))
        print("set_scaler = ", set_scaler)
        print(scaler)

        model = Earth()

        print("About to fit")
        model.fit(x_train, y_train)
        model.summary()
        pickle.dump(model, open('models/{}'.format(NAME), 'wb'))
        print("Model Saved")
        pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
        print("History Saved")
        #plot_model(model, show_shapes=True, show_layer_names=True)


    ########### Forward Pass ############################

    y_pred = (np.array(model.predict(x_test)).squeeze())
    y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()
    print(y_pred.dtype)

    print("y_test = ", y_test)
    print("y_pred = ", y_pred)





    #############################    INTEGRATION     ##########################
    int_naive, err_naive = cm.get_integral_uniform(y_test)
    int_MARS, err_MARS = cm.get_integral_uniform(y_pred)
    try:
        integ = vegas.Integrator(ndims* [[0, 1]])
        int_vegas = integ(f, nitn=10, neval=50000)

        print(int_vegas.summary())
        print('int_vegas = %s    Q = %.2f' % (int_vegas, int_vegas.Q))
        pull = cm.pull(int_vegas, int_MARS, err_MARS)
    except UnboundLocalError:
        print("f is not defined")
        int_vegas = 0
        pull = cm.pull(int_naive, int_MARS, err_MARS)

    #result_MARS = integ(MARS, nitn=10, neval=1000)


    #print(result_MARS.summary())

    print("Integral estimate with {} points".format(n_test))
    print('Naive Integral = {} +- {}'.format(int_naive, err_naive))
    print('MARS Integral = {} +- {}'.format(int_MARS, err_MARS))

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
        'log_acc':log_acc,
        'eta':eta,
        'alpha':alpha
    }

    csv_output = {
        'func':     func,
        'ndims':    [ndims],
        'set_scaler':set_scaler,
        'n_train':  [n_train],
        'n_test':   [n_test],
        'mse':      [mse],
        'mae':      [mae],
        'mape':     [mape],
        'int_MARS':   [int_MARS],
        'err_MARS':   [err_MARS],
        'int_naive':[int_naive],
        'int_vegas':[int_vegas],
        'pull':     [pull],
        'datetime': datetime.today().strftime('%Y-%m-%d-%H:%M:%S'),
        'r2':       [r2]
    }

    pd.DataFrame.from_dict(data=csv_output).to_csv('MARS_runs.csv', mode='a', header = False, index = False)

    #cm.plot_all(y_test, y_pred, measures, 'MARS', NAME)


if __name__ == '__main__':
    app.run(main)
