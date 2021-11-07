


import sys
sys.path.append('/home/chahrour/Interpolation/')
import itertools
import numpy as np
import time
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
from gpflow.ci_utils import ci_niter
from matplotlib.offsetbox import AnchoredText
from sklearn.model_selection import train_test_split
from Functions import Functions

import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
import common as cm
from datetime import datetime

import vegas
import pandas as pd

#plt.style.use("ggplot")

# for reproducibility of this notebook:
#rng = np.random.RandomState(123)
#tf.random.set_seed(42)

#tf.keras.backend.set_floatx('float32')
dtype = tf.float64
dtype_np = np.float64
gpflow.config.set_default_float(dtype)

#user_config = gpflow.config.Config(float=tf.float32, positive_bijector="exp")

from absl import app, flags


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
flags.DEFINE_integer('n_ind', 200 , '# of Inducing Points',
                     short_name = 'n_ind')
flags.DEFINE_float('lr', 1e-3, 'The learning rate',
                     short_name = 'lr')
flags.DEFINE_string('func', 'Gauss', 'Function',
                     short_name = 'f')
flags.DEFINE_string('scaler', '', 'Scaling/Normalization',
                     short_name = 's')

def run_adam(model, iterations, train_dataset, x_train, y_train, x_val, y_val,
            batch, lr):
    """
    Utility function running the Adam optimizer

    :param model: GPflow model
    :param interations: number of iterations
    """
    # Create an Adam Optimizer action
    logf = []
    mae_val = []
    mae_tr = []
    train_iter = iter(train_dataset.batch(batch))
    #print(np.shape(train_iter))
    #print(tf.shape(train_iter))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    #print((training_loss))
    optimizer = tf.keras.optimizers.Adam(lr)

    @tf.function
    def optimization_step():
        print(training_loss)
        print(model.trainable_variables)
        optimizer.minimize(training_loss, model.trainable_variables)

    for step in range(iterations):
        optimization_step()
        if step % 100 == 0:
            elbo = -training_loss().numpy()
            logf.append(elbo)
            y_pred_train, _ = model.predict_y(x_train)
            y_pred_val, _ = model.predict_y(x_val)
            mae_train = np.mean(np.abs(np.squeeze(y_pred_train) - y_train))
            mae_valid = np.mean(np.abs(np.squeeze(y_pred_val) - y_val))
            mae_val.append(mae_valid)
            mae_tr.append(mae_train)
            learning_rate = optimizer._decayed_lr('float32').numpy()
            print("epoch {}: elbo = {:.2f}, err_tr = {:.5f}, err_val = {:.5f}, lr = {:.5f}".format(step,
                    elbo, mae_train, mae_valid, learning_rate), flush=True)
    return logf, mae_tr, mae_val

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
    n_ind = FLAGS.n_ind
    set_scaler = FLAGS.scaler
    alpha = 0.2
    low = 0
    high = 1
    function = Functions(ndims, alpha)



    if func == "D3" or func == "D6" or func == "D9" or func == "Mxyzuv":
        _, _, _, x_train, y_train, x_test, y_test = cm.get_functions(func, ndims, alpha, n_train, n_test)
    else:
        (f, f_interp, integral_f) = cm.get_functions(func, ndims, alpha, n_train, n_test)
        x_train = np.random.rand(n_train, ndims)
        x_test = np.random.rand(n_test, ndims)
        y_test = f(x_test)
        y_train = f(x_train)
    ndims = x_train.shape[1]

    NAME = "{}_d_{}_e_{}_b_{}_ind_{}_opt_{}_lr_{}_{}_{:.0e}".format(func, ndims,
            epochs, batch, n_ind, opt, lr, set_scaler, n_train)
    # NAME = "{}_n_{}_l_{}_e_{}_b_{}_a_{}_l_{}_d_{}_opt_{}_{}_{}".format(func, nodes,
    #         layers, epochs, batch, activation, loss, ndims, opt, set_scaler,str(n_train))
    if func == 'Gauss':
        NAME = NAME + '_{:.1f}'.format(alpha)
    print(NAME)
    scaler = cm.get_scaler(set_scaler = set_scaler, low = low, high = high, load = load, NAME = NAME)


    y_train = (scaler.fit_transform(y_train.reshape(-1, 1))).ravel()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                        test_size=0.1, random_state=1)

    #x_train = np.random.rand(N, ndims) * 2 - 1  # x_train values
    #y_train = func(x_train)  # Noisy y_train values
    print("x_train = ", x_train)
    print("y_train = ", y_train)
    print("x_test = ", x_test)
    print("y_test = ", y_test)
    print("y_train max = ", np.max(y_train))
    print("y_train min = ", np.min(y_train))
    print("y_test max = ", np.max(y_test))
    print("y_test min = ", np.min(y_test))
    print(x_train.shape)
    print(y_train.shape)
    #x_train = np.random.rand(n_train, ndims)
    data = (x_train, y_train)


    if load:
        #m = tf.saved_model.load('models_svgp/{}'.format(NAME))
        m = tf.keras.models.load_model('models_svgp/{}'.format(NAME))
        #m = pickle.load(open('models_svgp/{}'.format(NAME), 'rb'))
        scaler = pickle.load(open('scaler_svgp/scaler_{}'.format(NAME), 'rb'))
        #logf = pickle.load(open('history/{}'.format(NAME), 'rb'))
        logf = pickle.load(open('history/Periodic_e_30000_b_1000_ind_2048_d_9_opt_adam_lr_0.001__5e+06', 'rb'))
        mae_tr = 0
        mae_val = 0
        print("Model has loaded")

    else:
        x_train = x_train.astype(dtype_np)
        y_train = y_train.astype(dtype_np)
        kernel = gpflow.kernels.SquaredExponential()
        #kernel = gpflow.kernels.Cosine()
        Z = x_train[:n_ind, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

        m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Gaussian(), Z, num_data=n_train)

        elbo = tf.function(m.elbo)


        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train[:,None])).repeat().shuffle(n_train)
        #train_dataset = tf.cast(train_dataset, dtype = tf.float32)
        print(np.shape(train_dataset))
        print(train_dataset)
        train_iter = iter(train_dataset.batch(batch))
        print(np.shape(train_iter))
        print(train_iter)

        # We turn off training for inducing point locations
        #gpflow.set_trainable(m.inducing_variable, False)

        maxiter = ci_niter(epochs)
        print(np.shape(maxiter))
        num_eval = 10**4
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=lr,
                        decay_steps=epochs//100,
                        decay_rate=0.99)
        logf, mae_tr, mae_val = run_adam(m, maxiter, train_dataset, x_train[:num_eval, ...],
                                        y_train[:num_eval], x_val[:num_eval, ...], y_val[:num_eval],
                                        batch, lr_schedule)
        m.predict_f_compiled = tf.function(
        m.predict_f, input_signature=[tf.TensorSpec(shape=[None, ndims], dtype=dtype)]
        )
        tf.saved_model.save(m, 'models/{}'.format(NAME))
        #pickle.dump(m, open('models_svgp/{}'.format(NAME), 'wb'))
        pickle.dump(scaler, open('scaler/scaler_{}'.format(NAME), 'wb'))
        pickle.dump(logf, open('history/{}'.format(NAME), 'wb'))

    # plt.plot(np.arange(maxiter)[::10], logf)
    # plt.xlabel("iteration")
    # _ = plt.ylabel("ELBO")
    # plt.savefig('plots_svgp/'+"ELBO_iteration.png")

    #plot("Predictions after training")
    start = time.time()
    y_pred, y_pred_v = m.predict_f_compiled(x_test)  # Predict y_train values at test locations
    end = time.time()
    time_pred = end - start
    print("Time taken for prediction = ", time_pred)
    y_pred = np.squeeze(y_pred)
    y_pred = np.array(scaler.inverse_transform(y_pred.reshape(-1, 1))).squeeze()

    mape = (y_test - y_pred)/y_test * 100
    print(np.shape(mape))
    # f, ax = plt.subplots(1,1)
    # ax.hist(mape, range=(-5, 5), bins = 100, histtype = 'step')
    # ax.set_yscale('log')
    # anchored_text = AnchoredText(r'$\epsilon_{{GP}}$ = {:0.2f} %'.format(np.mean(np.abs(mape))), loc=2)
    # ax.add_artist(anchored_text)
    # plt.savefig('plots_svgp/'+"MAPE.png")
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

    N = 10**5

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


    #########################################################

    measures = {
        'mse':mse,
        'mae':mae,
        'err':err,
        'sigfigs':sigfigs,
        'relerr':relerr,
        'mape':mape,
        'func':func,
        'loss_str':loss,
        'log_acc':log_acc,
        'eta':eta,
        'alpha':alpha,
        'logf':logf,
        'mae_tr':mae_tr,
        'mae_val':mae_val
    }

    csv_output = {
        'func':     func,
        'ndims':    [ndims],
        'n_ind':    [n_ind],
        'epochs':   [epochs],
        'batchsize':[batch],
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
        'datetime': datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
    }

    pd.DataFrame.from_dict(data=csv_output).to_csv('svgp_runs.csv', mode='a', header = False, index = False)

    #########################################################

    cm.plot_svgp(y_test, y_pred, measures, 'SVGP', NAME)

if __name__ == '__main__':
    app.run(main)
