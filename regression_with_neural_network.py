import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, max_error, explained_variance_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time


"""load data"""
if 1 == 1:
    ts0 = time.time()
    data_path = "D:\INDRA\ATM-TT-data\Datos de entrada"
    data = pd.read_csv(data_path + "data_set_airports.csv")

    print('Data loaded: ', round(time.time() - ts0, 2), 's')
    print('\n')


"""Preprocessing: encode categorical variables"""
if 1 == 1:
    labelEnc_local = LabelEncoder()
    labelEnc_airline = LabelEncoder()
    labelEnc_ades = LabelEncoder()

    Data = data.copy()
    Data.loc[:, 'LOCAL'] = labelEnc_local.fit_transform(data.loc[:, 'LOCAL'])
    Data.loc[:, 'AIRLINE'] = labelEnc_airline.fit_transform(data.loc[:, 'AIRLINE'])
    Data.loc[:, 'ADES IATA'] = labelEnc_ades.fit_transform(data.loc[:, 'ADES IATA'])

    X = Data.iloc[:, :-1]
    Y = Data.iloc[:, -1]

"""Train Test Split"""
if 1 == 1:
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    """Feature Scaling"""
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)


if 1 == 1:
    """ the max_iter=100 that you defined on the initializer is not in the grid. So, that number will be constant,
    while the ones in the grid will be searched"""
    nn = MLPClassifier(max_iter=100)

    """Define a hyper-parameter space to search. (All the values that you want to try out.)"""
    parameter_space = {
        'hidden_layer_sizes': [(50, 50, 50, 50), (50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }

    """ the parameter n_jobs is to define how many CPU cores from your computer to use
     (-1 is for all the cores available). The cv is the number of splits for cross-validation."""
    clf = GridSearchCV(nn, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X_train, y_train)

    """see results"""
    """Best parameters set"""
    print('Best parameters found:\n', clf.best_params_)

    """All results"""
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))



"""Train models"""
if 1 == 1:

    """Neural Network"""
    if 1 == 1:
        ts1 = time.time()
        nn = MLPClassifier(hidden_layer_sizes=(20, 40, 40, 20),
                           max_iter=1000)
        nn.fit(X_train, y_train.values.ravel())
        print('Neural Network trained:', round(time.time() - ts1, 2), 's')
        print('\n')

    """Random forest"""
    if 1 == 1:
        ts2 = time.time()
        rf = RandomForestRegressor(n_estimators=50,
                                   random_state=42)
        rf.fit(X_train, y_train.values.ravel())
        print('Random Forest trained:', round(time.time() - ts2, 2), 's')
        print('\n')


"""Make predictions"""
if 1 == 1:
    y_hat_rf = rf.predict(X_test)
    y_hat_nn = nn.predict(X_test)

"""Evaluate models with Regression metrics"""
if 1 == 1:
    def obtain_regression_metrics(y_hat, y_test, algorithm_name):
        print('===' + algorithm_name + '===')
        print('R2::', r2_score(y_test, y_hat))
        print('MSE:', mean_squared_error(y_test, y_hat))
        print('MAE:', mean_absolute_error(y_test, y_hat))
        print('MAX ERROR:', max_error(y_test, y_hat))
        print('EXPLANINED VARIANCE:', explained_variance_score(y_test, y_hat))
        print('\n')


    obtain_regression_metrics(y_hat_nn, y_test, 'Neural Network')
    obtain_regression_metrics(y_hat_rf, y_test, 'Random Forest')
