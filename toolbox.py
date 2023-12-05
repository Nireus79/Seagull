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


def spliter(full_data, research_data, signal, part):
    """
    spliter takes a full dataset, and a dataset containing only cases for training and testing.
    drops the column of returns if classification is researched
    or bins if regression is researched.
    Then splits the research_data into X (features) and Y(labels),
    drops 'Open', 'High', 'Low', 'Close', 'Volume' as those needed only into the backtest_data for use in bt.py lib
    Then splits X and Y for training and testing by 0.8 and 0.2 according to arg given part.
    :param full_data:
    :param research_data: dataset containing only cases for training and testing
    :param signal:
    :param part: 1 to 5
    :return: X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    """
    Y = research_data.loc[:, signal]
    Y.name = Y.name
    X = research_data.loc[:, research_data.columns != signal, ]
    Y = research_data.loc[:, Y.name]
    X = research_data.loc[:, X.columns]
    if signal == 'ret':
        X.drop(columns=['Close', 'Open', 'High', 'Low', 'Volume', 'Volatility', 'Dema9', '4H%K', '4H%D', '4H%DS',
                        'bb_cross', 'bin'],
               axis=1, inplace=True)
    elif signal == 'bin':
        X.drop(columns=['Close', 'Open', 'High', 'Low', 'Volume', 'Dema9', 'Volatility', '4H%K', 'momentum',
                        'elder', 'bb_cross', 'ret'],
               axis=1, inplace=True)
    validation_size = 0.2
    test_size = int(len(X) * validation_size)
    if part == 1:
        X_test, X_train = X[:test_size], X[test_size:]
        Y_test, Y_train = Y[:test_size], Y[test_size:]
        backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
        return X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    elif part == 2:
        X_test, X_train = X[test_size:test_size * 2], pd.concat([X[:test_size], X[test_size * 2:]])
        Y_test, Y_train = Y[test_size:test_size * 2], pd.concat([Y[:test_size], Y[test_size * 2:]])
        backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
        return X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    elif part == 3:
        X_test, X_train = X[test_size * 2:test_size * 3], pd.concat([X[:test_size * 2], X[test_size * 3:]])
        Y_test, Y_train = Y[test_size * 2:test_size * 3], pd.concat([Y[:test_size * 2], Y[test_size * 3:]])
        backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
        return X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    elif part == 4:
        X_test, X_train = X[test_size * 3:test_size * 4], pd.concat([X[:test_size * 3], X[test_size * 4:]])
        Y_test, Y_train = Y[test_size * 3:test_size * 4], pd.concat([Y[:test_size * 3], Y[test_size * 4:]])
        backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
        return X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    elif part == 5:
        X_test, X_train = X[test_size * 4:], X[:test_size * 4]
        Y_test, Y_train = Y[test_size * 4:], Y[:test_size * 4]
        backtest_data = full_data[X_test.index[0]:X_test.index[-1]]
        return X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    else:
        print('Give part number 1 to 5 only.')


def meta_spliter(full_data, research_data, signal, part):
    """
    spliter takes a full dataset, and a dataset containing only cases for training and testing.
    drops the column of returns if classification is researched
    or bins if regression is researched.
    Then splits the research_data into X (features) and Y(labels),
    drops 'Open', 'High', 'Low', 'Close', 'Volume' as those needed only into the backtest_data for use in bt.py lib
    Then splits X and Y for training and testing by 0.8 and 0.2 according to arg given part.
    :param full_data:
    :param research_data: dataset containing only cases for training and testing
    :param signal:
    :param part: 1 to 5
    :return: X, Y, X_train, X_test, Y_train, Y_test, backtest_data
    """
    train_size = 0.8
    train_split = 0.5
    if part == 5:
        train_set = research_data[:int(len(research_data) * train_size)]
        research_data1 = train_set[:int(len(train_set) * train_split)]
        research_data2 = train_set[int(len(train_set) * train_split):]
        test_data = research_data[int(len(research_data) * train_size):]

        Y1 = research_data1.loc[:, signal]
        Y1.name = Y1.name
        X1 = research_data1.loc[:, research_data1.columns != signal]
        Y1 = research_data1.loc[:, Y1.name]
        X1 = research_data1.loc[:, X1.columns]

        Y2 = research_data2.loc[:, signal]
        Y2.name = Y2.name
        X2 = research_data2.loc[:, research_data2.columns != signal]
        Y2 = research_data2.loc[:, Y2.name]
        X2 = research_data2.loc[:, X2.columns]

        Y3 = test_data.loc[:, signal]
        Y3.name = Y3.name
        X3 = test_data.loc[:, test_data.columns != signal]
        Y3 = test_data.loc[:, Y3.name]
        X3 = test_data.loc[:, X3.columns]
        if signal == 'ret':
            X1.drop(columns=['bin'], axis=1, inplace=True)
            X2.drop(columns=['bin'], axis=1, inplace=True)
            X3.drop(columns=['bin'], axis=1, inplace=True)
        elif signal == 'bin':
            X1.drop(columns=['ret'], axis=1, inplace=True)
            X2.drop(columns=['ret'], axis=1, inplace=True)
            X3.drop(columns=['ret'], axis=1, inplace=True)
        backtest_data = full_data[X3.index[0]:]
        return X1, Y1, X2, Y2, X3, Y3, backtest_data


def evaluate_arima_model(X_train, Y_train, arima_order):
    """
    evaluate an ARIMA model for a given order (p,d,q)
    Assuming that the train and Test Data is already defined before
    :param Y_train:
    :param X_train:
    :param arima_order: (p,d,q)
    :return: mean_squared_error
    """
    # predicted = list()
    modelARIMA = ARIMA(endog=Y_train, exog=X_train, order=arima_order)
    model_fit = modelARIMA.fit()
    error = mean_squared_error(Y_train, model_fit.fittedvalues)
    return error


#
def evaluate_arima_models(X_train, Y_train, p_values, d_values, q_values):
    """
    evaluate combinations of p, d and q values for an ARIMA model
    :param Y_train:
    :param X_train:
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
                mse = evaluate_arima_model(X_train, Y_train, order)
                if mse < best_score:
                    best_score, best_cfg = mse, order
                print('ARIMA%s MSE=%.7f' % (order, mse))
    print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))


def create_LSTMmodel(X_train, neurons, learn_rate, momentum):
    # create model
    mdl = Sequential()
    mdl.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
    # Number of cells can be added if needed
    # mdl.add(Dense(1))
    # mdl.add(Dense(1))
    # mdl.add(Dense(1))
    # mdl.add(Dense(1))
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    mdl.add(Dense(1))
    # optimizer = SGD(learning_rate=learn_rate, momentum=momentum)
    mdl.compile(loss='mse', optimizer='adam')
    return mdl


def evaluate_LSTM(X_train, X_test, Y_test, nr, lrn, mom):
    LSTM_Model = create_LSTMmodel(X_train, nr, lrn, mom)
    pred = LSTM_Model.predict(X_test)
    error = mean_squared_error(pred, Y_test)
    return error


def evaluate_LSTM_combinations(X_train, X_test, Y_test, neurons_list, learn_rate_list, momentum_list):
    best_score, best_cfg = float('inf'), None
    for n in neurons_list:
        for lr in learn_rate_list:
            for m in momentum_list:
                combination = (n, lr, m)
                mse = evaluate_LSTM(X_train, X_test, Y_test, n, lr, m)
                if mse < best_score:
                    best_score, best_cfg = mse, combination
                print('LSTM: {} mse: {}'.format(combination, mse))
    print('Best LSTM: {} mse: {}'.format(best_cfg, best_score))


def create_ANN(X_train, Y_train, units=6, batch=10, epochs=100):
    # Initialization
    model = Sequential()
    # Input layer
    model.add(Dense(units=units, kernel_initializer='uniform', activation='relu', input_dim=4))
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
    model.fit(X_train, Y_train, batch_size=batch, epochs=epochs)


def evaluate_ANN(X_train, Y_train, X_test, Y_test, units, batch, epochs):
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
                model.fit(X_train, Y_train, batch_size=b, epochs=e)
                # Evaluation
                y_pred = model.predict(X_test)
                y_pred = pd.DataFrame(y_pred)
                print(y_pred.describe())
                scores = model.evaluate(X_test, Y_test)
                print(model.metrics_names[1], scores[1])


def ROC(df, n):
    M = df.diff(n - 1)
    N = df.shift(n - 1)
    roc = pd.Series(((M / N) * 100), name='ROC_' + str(n))
    return roc


def MOM(df, n):
    mom = pd.Series(df.diff(n), name='Momentum_' + str(n))
    return mom


def crossing2(df, col1, col2, col3):
    crit1 = df[col2].shift(1) < df[col3].shift(1)
    crit2 = df[col2] > df[col3]
    up_cross = df[col1][crit1 & crit2]
    side_up = pd.Series(1, index=up_cross.index)

    crit3 = df[col2].shift(1) > df[col3].shift(1)
    crit4 = df[col2] < df[col3]
    down_cross = df[col1][crit3 & crit4]
    side_down = pd.Series(-1, index=down_cross.index)

    return pd.concat([side_up, side_down]).sort_index()


def crossing3(df, col1, col2, col3):
    crit1 = df[col1].shift(1) < df[col2].shift(1)
    crit2 = df[col1] > df[col2]
    up_cross = df[col1][crit1 & crit2]
    side_up = pd.Series(1, index=up_cross.index)

    crit3 = df[col1].shift(1) > df[col3].shift(1)
    crit4 = df[col1] < df[col3]
    down_cross = df[col1][crit3 & crit4]
    side_down = pd.Series(-1, index=down_cross.index)

    return pd.concat([side_up, side_down]).sort_index()
