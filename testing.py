import winsound
import numpy as np
from backtesting import Backtest, Strategy
from data_forming import data, X_train, Y_train
from sklearn.linear_model import LinearRegression

data['X_Close'] = data['Close']
data.rename(columns={'volatility': 'X_volatility', 'etheur_close': 'X_etheur_close', 'ema9': 'X_ema9'}, inplace=True)


def get_X(dt):
    """Return model design matrix X"""
    return dt.filter(like='X').values


validation_size = 0.2
train_size = int(len(data) * (1 - validation_size))
train_data, test_data = data[0:train_size], data[train_size:len(data)]
print(test_data)


# data = data.fillna(0)
# print(test_data)
# print(test_data.isnull().sum())
# print('total ret', np.sum(np.array(test_data.ret) != 0, axis=0))
# print('positive ret', np.sum(np.array(test_data.ret) > 0, axis=0))
# print('negative ret', np.sum(np.array(test_data.ret) < 0, axis=0))


class Seagull(Strategy):
    state = None

    def init(self):
        self.model = LinearRegression()
        self.model.fit(X_train, Y_train)
        # Prepare empty, all-NaN forecast indicator
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)))

        # self.L_atr = self.I(atr_wrapper, self.data.df)
        # self.M_D = self.I(sma_indicator, pd.Series(self.M_K), 3, fillna=False)

    def next(self):
        ret = self.data.ret
        X = get_X(self.data)
        forecast = self.model.predict(X)[0]
        self.forecasts[-1] = forecast

        if self.position.is_long and ret > 0 and forecast > 0:
            print(ret, forecast, self.data.doteur_close)
            self.buy()
        elif self.position.is_short and ret < 0 and forecast < 0:
            print(ret, forecast, self.data.doteur_close)
            self.sell()


def statistics():
    bt = Backtest(test_data, Seagull, cash=100000, commission=0.004, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt():
    bt = Backtest(test_data, Seagull, cash=100000, commission=0.004, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        # ema=range(5, 31, 1),
        # rs=range(50, 76, 1),
        # upper_bound=range(30, 100, 1),
        lower_bound=range(5, 35, 1),
        # maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    winsound.Beep(1000, 1500)


statistics()

# opt()
