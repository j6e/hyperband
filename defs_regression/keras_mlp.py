"function (and parameter space) definitions for hyperband"
"regression with Keras (multilayer perceptron)"

from common_defs import *

# a dict with x_train, y_train, x_test, y_test
#from load_data_for_regression import data

import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import *

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler

#

# TODO: advanced activations - 'leakyrelu', 'prelu', 'thresholdedrelu', 'srelu'
# TODO: Regularizations L2, L1, L1_L2 and none


max_layers = 3
max_layer_size = 100
iters_mult = 4

space = {
    'scaler': hp.choice('s',
                        (None, 'StandardScaler', 'RobustScaler', 'MinMaxScaler', 'MaxAbsScaler')),
    'n_layers': hp.quniform('ls', 1, max_layers, 1),
    # 'layer_size': hp.quniform( 'ls', 5, 100, 1 ),
    # 'activation': hp.choice( 'a', ( 'relu', 'sigmoid', 'tanh' )),
    'init': hp.choice('i', ('uniform', 'normal', 'glorot_uniform',
                            'glorot_normal', 'he_uniform', 'he_normal')),
    'batch_size': hp.choice('bs', (1, 4, 8, 10, 16, 32, 50, 64, 100, 128, 200, 256)),
    'shuffle': hp.choice('sh', (False, True)),
    'loss': hp.choice('l', ('mean_absolute_error', 'mean_squared_error')),
    'optimizer': hp.choice('o', ('rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'sgd'))
}

# for each hidden layer, we choose size, activation and extras individually
for i in range(1, max_layers + 1):
    space['layer_{}_size'.format(i)] = hp.quniform('ls{}'.format(i),
                                                   2, max_layer_size, 1)
    space['layer_{}_activation'.format(i)] = hp.choice('a{}'.format(i),
                                                       ('relu', 'sigmoid', 'tanh', 'elu', 'selu'))
    space['layer_{}_kernel_regularization'.format(i)] = hp.choice('r{}'.format(i),
                                                            ({'name': 'l1_l2',
                                                              'rate': hp.choice('kr_l1l2_{}'.format(i), (0.1, 0.01, 0.001, 0.))},
                                                             {'name': 'l2',
                                                              'rate': hp.choice('kr_l2_{}'.format(i), (0.1, 0.01, 0.001, 0.))},
                                                             {'name': 'l1',
                                                              'rate': hp.choice('kr_l1_{}'.format(i), (0.1, 0.01, 0.001, 0.))}))
    space['layer_{}_activity_regularization'.format(i)] = hp.choice('r{}'.format(i),
                                                            ({'name': 'l1_l2',
                                                              'rate': hp.choice('ar_l1l2_{}'.format(i), (0.1, 0.01, 0.001, 0.))},
                                                             {'name': 'l2',
                                                              'rate': hp.choice('ar_l2_{}'.format(i), (0.1, 0.01, 0.001, 0.))},
                                                             {'name': 'l1',
                                                              'rate': hp.choice('ar_l1_{}'.format(i), (0.1, 0.01, 0.001, 0.))}))

    space['layer_{}_extras'.format(i)] = hp.choice('e{}'.format(i), (
        {'name': 'dropout', 'rate': hp.uniform('d{}'.format(i), 0.05, 0.5)},
        {'name': 'batchnorm'},
        {'name': None}
    ))


def get_params():
    params = sample(space)
    return handle_integers(params)


#

# print hidden layers config in readable way
def print_layers(params):
    for i in range(1, params['n_layers'] + 1):
        print("layer {} | size: {:>3} | activation: {:<7} | act_reg: {:5}({:.3f}) | ker_reg: {:5}({:.3f}) | extras: {}".format(i,
                                                            params['layer_{}_size'.format(i)],
                                                            params['layer_{}_activation'.format(i)],
                                                            params['layer_{}_kernel_regularization'.format(i)]['name'],
                                                            params['layer_{}_kernel_regularization'.format(i)]['rate'],
                                                            params['layer_{}_activity_regularization'.format(i)]['name'],
                                                            params['layer_{}_kernel_regularization'.format(i)]['rate'],
                                                            params['layer_{}_extras'.format(i)]['name']), end=' ')
        if params['layer_{}_extras'.format(i)]['name'] == 'dropout':
            print("- rate: {:.1%}".format(params['layer_{}_extras'.format(i)]['rate']), end=' ')
        print()


def print_params(params):
    pprint({k: v for k, v in list(params.items()) if not k.startswith('layer_')})
    print_layers(params)
    print()


def _get_regularizations(params, layer):
    if params['layer_{}_kernel_regularization'.format(layer)]['name']:
        k_reg = eval('regularizers.{}({})'.format(params['layer_{}_kernel_regularization'.format(layer)]['name'],
                                                  params['layer_{}_kernel_regularization'.format(layer)]['rate']))
    else:
        k_reg = None

    if params['layer_{}_activity_regularization'.format(layer)]['name']:
        a_reg = eval('regularizers.{}({})'.format(params['layer_{}_activity_regularization'.format(layer)]['name'],
                                                  params['layer_{}_activity_regularization'.format(layer)]['rate']))
    else:
        a_reg = None

    return k_reg, a_reg


def try_params(n_iterations, params, data, return_model=False, early_stop=True):
    n_iterations = int(n_iterations * iters_mult)
    print("iterations:", n_iterations)
    print_params(params)

    y_train = data['y_train']
    y_test = data['y_test']

    if params['scaler']:
        scaler_x = eval("{}()".format(params['scaler']))
        x_train_ = scaler_x.fit_transform(data['x_train'].astype(float))
        x_test_ = scaler_x.transform(data['x_test'].astype(float))

        scaler_y = eval("{}()".format(params['scaler']))
        y_train = scaler_y.fit_transform(data['y_train'].reshape(-1, 1).astype(float))
        y_test = scaler_y.transform(data['y_test'].reshape(-1, 1).astype(float))
    else:
        x_train_ = data['x_train']
        x_test_ = data['x_test']

    input_dim = x_train_.shape[1]

    k_reg, a_reg = _get_regularizations(params, 1)

    model = Sequential()
    model.add(Dense(params['layer_1_size'], kernel_initializer=params['init'],
                    activation=params['layer_1_activation'], input_dim=input_dim,
                    kernel_regularizer=k_reg, activity_regularizer=a_reg))
    last = 1

    for i in range(int(params['n_layers']) - 1):

        extras = 'layer_{}_extras'.format(i + 1)

        if params[extras]['name'] == 'dropout':
            model.add(Dropout(params[extras]['rate']))
        elif params[extras]['name'] == 'batchnorm':
            model.add(BatchNorm())

        k_reg, a_reg = _get_regularizations(params, i + 2)

        model.add(Dense(params['layer_{}_size'.format(i + 2)], kernel_initializer=params['init'],
                        activation=params['layer_{}_activation'.format(i + 2)],
                        kernel_regularizer=k_reg, activity_regularizer=a_reg))
        last = i + 2

    extras = 'layer_{}_extras'.format(last)
    if params[extras]['name'] == 'dropout':
        model.add(Dropout(params[extras]['rate']))
    elif params[extras]['name'] == 'batchnorm':
        model.add(BatchNorm())

    model.add(Dense(1, kernel_initializer=params['init'], activation='linear'))
    model.compile(optimizer=params['optimizer'], loss=params['loss'])

    #print(model.summary())

    #

    validation_data = (x_test_, y_test)

    if early_stop:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    else:  # Never stop...
        early_stopping = EarlyStopping(monitor='train_loss', patience=10000, verbose=0)

    history = model.fit(x_train_, y_train,
                        epochs=int(round(n_iterations)),
                        batch_size=params['batch_size'],
                        shuffle=params['shuffle'],
                        validation_data=validation_data,
                        callbacks=[early_stopping])

    #
    p = model.predict(x_train_, batch_size=params['batch_size'])
    p = np.nan_to_num(p)

    if params['scaler']:
        p = scaler_y.inverse_transform(p)
        y_train = scaler_y.inverse_transform(y_train)

    mse = MSE(y_train, p)
    rmse = sqrt(mse)
    mae = MAE(y_train, p)

    print("\n# training | RMSE: {:.4f}, MAE: {:.4f}".format(rmse, mae))

    #
    p = model.predict(x_test_, batch_size=params['batch_size'])
    p = np.nan_to_num(p)
    if params['scaler']:
        p = scaler_y.inverse_transform(p)
        y_test = scaler_y.inverse_transform(y_test)

    mse = MSE(y_test, p)
    rmse = sqrt(mse)
    mae = MAE(y_test, p)

    print("# testing  | RMSE: {:.4f}, MAE: {:.4f}".format(rmse, mae))
    if return_model:
        return model
    else:
        return {'loss': rmse, 'rmse': rmse, 'mae': mae, 'early_stop': model.stop_training}
