from abc import ABC

import winsound
import numpy as np
from backtesting import Backtest, Strategy
from data_forming import full_data, research_data, backtest_data, X_train, Y_train, X_test, Y_test, eth, threshold
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings('ignore')


# data = data.fillna(0)
# print(test_data)
# print(test_data.isnull().sum())
# print('total ret', np.sum(np.array(test_data.ret) != 0, axis=0))
# print('positive ret', np.sum(np.array(test_data.ret) > 0, axis=0))
# print('negative ret', np.sum(np.array(test_data.ret) < 0, axis=0))

# X_train_ARIMA = X_train.loc[:, ['Close', 'Dema13', '4H%K', 'Volatility', 'Etherium']]


class Seagull(Strategy):

    def init(self):
        # self.model = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=[2, 1, 1])
        # self.model_fit = self.model.fit()
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X_train, Y_train)
        self.buy_price = 0
        self.sell_price = 0
        self.commission = 4.5

    def next(self):
        ret = self.data.ret[-1]
        forecast = self.model.predict([[self.data['4H_rsi'][-1], self.data['4H%K'][-1],
                                        self.data['4H%D'][-1], self.data['4Hmacd'][-1]]])
        if not self.position.is_long and ret != 0 and forecast / self.data.Close[-1] > self.commission:
            self.buy_price = self.data.Close[-1]
            backtest_data['b'].loc[self.data.index[-1]] = True
            self.buy()
        elif not self.position.is_short and self.data.Close < self.buy_price and\
                forecast / self.data.Close[-1] < self.commission:
            self.sell_price = self.data.Close[-1]
            print(self.data.index[-1], 'Pillow profit: ', self.sell_price - self.buy_price)
            self.sell_price, self.buy_price = 0, 0
            backtest_data['s'].loc[self.data.index[-1]] = True
            self.sell()
        elif not self.position.is_short and ret != 0 and forecast / self.data.Close[-1] < self.commission:
            self.sell_price = self.data.Close[-1]
            print(self.data.index[-1], 'Trade profit: ', self.sell_price - self.buy_price)
            self.sell_price, self.buy_price = 0, 0
            backtest_data['s'].loc[self.data.index[-1]] = True
            self.sell()


def statistics():
    bt = Backtest(backtest_data, Seagull, cash=100000, commission=0.045, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt():
    bt = Backtest(backtest_data, Seagull, cash=100000, commission=0.045, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        # ema=range(5, 31, 1),
        # rs=range(50, 76, 1),
        # upper_bound=range(30, 100, 1),
        commission=range(0, 5, 1),
        # maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    # winsound.Beep(1000, 1500)


backtest_data['b'] = 0
backtest_data['s'] = 0
statistics()
# opt()

# print(backtest_data)
# print(backtest_data.b.sum())
# print(backtest_data.s.sum())
# opt()
