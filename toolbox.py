import pandas as pd
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM


def data_merger(path):
    # path = "E:/T/ETHUSDT/10mdb/"  # set this to the folder containing CSVs
    names = glob.glob(path + "*.csv")  # get names of all CSV files under path
    # If your CSV files use commas to split fields, then the sep
    # argument can be ommitted or set to ","
    file_list = pd.concat([pd.read_csv(filename, sep=",", header=None) for filename in names])
    # save the DataFrame to a file
    # csv = file_list.to_csv("sample.csv")
    return file_list


def primary_asset_merger(csv_path):
    merged = data_merger(csv_path)
    merged.columns = ['time', 'Open', 'High', 'Low', 'Close', 'Volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                      'ignore']
    # merged.time = pd.to_datetime(merged.time, unit='ms')
    merged.set_index('time', inplace=True)
    merged.drop(['close_time', 'quote_asset_volume', 'number_of_trades',
                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)
    merged = merged[~merged.index.duplicated(keep='first')]
    # print(merged)
    return merged


def asset_merger(csv_path, asset):
    merged = data_merger(csv_path)
    merged.columns = ['time', asset + '_open', asset + '_high', asset + '_low', asset + '_close', 'volume',
                      'close_time', 'quote_asset_volume', 'number_of_trades',
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                      'ignore']
    # merged.time = pd.to_datetime(merged.time, unit='ms')
    merged.set_index('time', inplace=True)
    merged.drop(['close_time', 'quote_asset_volume', 'number_of_trades',
                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'], axis=1, inplace=True)
    merged = merged[~merged.index.duplicated(keep='first')]
    # print(merged)
    return merged
    # merged.to_csv('dot-eur20202023hours.csv')


def rescaler(data, minmax):
    scaler = MinMaxScaler(feature_range=minmax)
    rescaled = pd.DataFrame(scaler.fit_transform(data), index=data.index)
    rescaled.columns = data.columns
    return rescaled


def standardizer(data):
    scaler = StandardScaler().fit(data)
    standardized = pd.DataFrame(scaler.fit_transform(data), index=data.index)
    standardized.columns = data.columns
    # print('standardizedX -----')
    # print(standardized.head())
    return standardized


def normalizer(data):
    scaler = Normalizer().fit(data)
    normalized = pd.DataFrame(scaler.fit_transform(data), index=data.index)
    normalized.columns = data.columns
    # print('normalizedX -----')
    # print(normalized.head())
    return normalized


def evaluate_arima_model(Xt, Yt, arima_order):
    """
    evaluate an ARIMA model for a given order (p,d,q)
    Assuming that the train and Test Data is already defined before
    :param Yt:
    :param Xt:
    :param arima_order: (p,d,q)
    :return: mean_squared_error
    """
    # predicted = list()
    modelARIMA = ARIMA(endog=Yt, exog=Xt, order=arima_order)
    model_fit = modelARIMA.fit()
    error = mean_squared_error(Yt, model_fit.fittedvalues)
    return error


#
def evaluate_arima_models(Xtr, Ytr, p_values, d_values, q_values):
    """
    evaluate combinations of p, d and q values for an ARIMA model
    :param Ytr: Y training
    :param Xtr: X training
    :param p_values:
    :param d_values:
    :param q_values:
    :return: best combo
    """
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                mse = evaluate_arima_model(Xtr, Ytr, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.7f' % (order, mse))
    print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))


def create_LSTMmodel(Xtr, neurons, learn_rate, momentum):
    # create model
    mdl = Sequential()
    mdl.add(LSTM(neurons, input_shape=(Xtr.shape[1], Xtr.shape[2])))
    # Number of cells can be added if needed
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    mdl.compile(loss='mse', optimizer='adam')
    return mdl


def evaluate_LSTM(Xtr, Xts, Yts, nr, lrn, mom):
    LSTM_Model = create_LSTMmodel(Xtr, nr, lrn, mom)
    pred = LSTM_Model.predict(Xts)
    error = mean_squared_error(pred, Yts)
    return error


def evaluate_LSTM_combinations(Xtr, Xts, Yts, neurons_list, learn_rate_list, momentum_list):
    best_score, best_cfg = float('inf'), None
    for n in neurons_list:
        for lr in learn_rate_list:
            for m in momentum_list:
                combination = (n, lr, m)
                mse = evaluate_LSTM(Xtr, Xts, Yts, n, lr, m)
                if mse < best_score:
                    best_score, best_cfg = mse, combination
                print('LSTM: {} mse: {}'.format(combination, mse))
    print('Best LSTM: {} mse: {}'.format(best_cfg, best_score))


def create_ANN(X_tr, Y_tr, units=6, batch=10, epochs=100):
    # Initialization
    model = Sequential()
    # Input layer
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim=5))
    # Hidden layers
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu'))
    # Output layer
    model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    # Compilation
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # Fitting
    model.fit(X_tr, Y_tr, batch_size=batch, epochs=epochs)


def evaluate_ANN(X_tr, Y_tr, X_tst, Y_tst, units, batch, epochs):
    for u in units:
        for b in batch:
            for e in epochs:
                # Initialization
                model = Sequential()
                # Input layer
                model.add(Dense(units=u, kernel_initializer='uniform', activation='relu', input_dim=5))
                # Hidden layers
                model.add(Dense(units=u, kernel_initializer='uniform', activation='relu'))
                model.add(Dense(units=u, kernel_initializer='uniform', activation='relu'))
                model.add(Dense(units=u, kernel_initializer='uniform', activation='relu'))
                model.add(Dense(units=u, kernel_initializer='uniform', activation='relu'))
                model.add(Dense(units=u, kernel_initializer='uniform', activation='relu'))
                # Output layer
                model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
                # Compilation
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                # Fitting
                model.fit(X_tr, Y_tr, batch_size=b, epochs=e)
                # Evaluation
                y_pred = model.predict(X_tst)
                y_pred = pd.DataFrame(y_pred)
                print(y_pred.describe())
                scores = model.evaluate(X_tst, Y_tst)
                print(model.metrics_names[1], scores[1])


def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    ROC = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return ROC


def MOM(df, n):
    MOM = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return MOM
