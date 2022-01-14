# https://scikit-learn.org/stable/modules/neural_networks_supervised.html

import numpy as np
import matplotlib.pyplot as pl
from sklearn import neural_network
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# []
X = [[0., 0.], [1., 1.]]
dataY = pd.read_csv('losses.csv')
dataX1 = pd.read_csv('demand.csv')
dataX2 = pd.read_csv('generation.csv') * 4000

y = np.zeros((dataY['time'].unique().size, 1))
x1 = np.zeros((dataX1['time'].unique().size, 1))
x2 = dataX2.gen.values.reshape(144, 1)
j = 0
for i in dataY['time'].unique():
    y[j] = dataY[dataY['time'] == i].plosses.sum()
    x1[j] = dataX1[dataX1['time'] == i].Pd.sum()
    j = j + 1

# X = np.concatenate((x1,x2),axis=1)
scaler = MinMaxScaler()

scaler.fit(x1)
X = scaler.transform(x1)

scaler = MinMaxScaler()
scaler.fit(y)
Y = scaler.transform(y)

coef = np.random.rand()
n, m = X.shape
cv = round(0.8 * n)
X_train = X
X_test = np.dot(X, abs(np.random.normal(0, 0.1, size=(m, m))))

Y_train = Y
Y_test = np.dot(Y, abs(np.random.normal(0, 0.1)))

if 0 == 1:
    plt.figure()
    plt.plot(X_train[:, 0], 'k--')
    plt.plot(X_train[:, 1], 'k--')
    plt.plot(X_test[:, 0], 'k-')
    plt.plot(X_test[:, 1], 'k-')
    plt.show()

    plt.figure()
    plt.plot(Y_train, 'r-')
    plt.plot(Y_test, 'r-')
    plt.show()

if 0 == 1:
    mlp = MLPRegressor(activation='logistic',  # activation function
                       alpha=1e-05,  # L2 penalty (regularization term) parameter
                       batch_size='auto',
                       beta_1=0.9,
                       beta_2=0.999,
                       early_stopping=False,
                       epsilon=1e-08,
                       hidden_layer_sizes=(7,),  # The ith element represents the number of neurons in the ith hidden layer
                       learning_rate='constant',  # The initial learning rate
                       learning_rate_init=0.001,
                       max_iter=200,
                       momentum=0.9,
                       n_iter_no_change=10,  # Maximum number of epochs
                       nesterovs_momentum=True,
                       power_t=0.5,
                       random_state=1,
                       shuffle=True,
                       solver='sgd',
                       tol=0.0001,
                       validation_fraction=0.1,
                       verbose=False,
                       warm_start=False)
if 1 == 1:
    mlp = MLPRegressor(
    )
    mlp.fit(X_train, Y_train)
    # For architecture 56:25:11:7:5:3:1 with input 56 and 1 output hidden layers will be (25:11:7:5:3). So tuple hidden_layer_sizes = (25,11,7,5,3,)
    # For architecture 3:45:2:11:2 with input 3 and 2 output hidden layers will be (45:2:11). So tuple hidden_layer_sizes = (45,2,11,)
    Y_predict = mlp.predict(X_test)
    APE = abs(Y_predict.reshape((Y_predict.size, 1)) - Y_test)

    plt.figure()
    plt.plot(Y_predict, 'r-')
    plt.plot(Y_test, 'r-')
    plt.show()

    plt.figure()
    plt.plot(X_train, 'k-')
    plt.show()

    plt.figure()
    plt.plot(APE, 'b-')
    plt.show()

    # print(mlp.score([[2., 2.], [1., 2.]], y = [0, 1]))

if 0 == 1:
    # MLP Regression 10-fold CV example using Boston dataset.
    # ###########################################
    # Load data
    boston = datasets.load_boston()
    # Creating Regression Design Matrix
    x = boston.data
    # Creating target dataset
    y = boston.target
    # x_train, x_test, y_train, y_test= cross_validation.train_test_split(x, y, test_size=0.2, random_state=42)
    # ######################################################################
    # Fit regression model
    n_fig = 0
    for name, nn_unit in [
        ('MLP using ReLU', neural_network.MLPRegressor(activation='relu', solver='lbfgs')),
        ('MLP using Logistic Neurons', neural_network.MLPRegressor(activation='logistic')),
        ('MLP using TanH Neurons', neural_network.MLPRegressor(activation='tanh', solver='lbfgs'))
    ]:
        regressormodel = nn_unit.fit(x, y)
        # Y predicted values
        yp = nn_unit.predict(x)
        rmse = np.sqrt(mean_squared_error(y, yp))
        # Calculation 10-Fold CV
        yp_cv = cross_val_predict(regressormodel, x, y, cv=10)
        rmsecv = np.sqrt(mean_squared_error(y, yp_cv))
        print('Method: %s' % name)
        print('RMSE on the data: %.4f' % rmse)
        print('RMSE on 10-fold CV: %.4f' % rmsecv)
        n_fig = n_fig + 1
        pl.figure(n_fig)
        pl.plot(yp, y, 'ro')
        pl.plot(yp_cv, y, 'bo', alpha=0.25, label='10-folds CV')
        pl.xlabel('predicted')
        pl.title('Method: %s' % name)
        pl.ylabel('real')
        pl.grid(True)
        pl.show()
