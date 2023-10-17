import winsound
import numpy as np
from backtesting import Backtest, Strategy
from data_forming import full_data, research_data, backtest_data, X_train, Y_train, X_test, Y_test
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# data = data.fillna(0)
# print(test_data)
# print(test_data.isnull().sum())
# print('total ret', np.sum(np.array(test_data.ret) != 0, axis=0))
# print('positive ret', np.sum(np.array(test_data.ret) > 0, axis=0))
# print('negative ret', np.sum(np.array(test_data.ret) < 0, axis=0))


class Seagull(Strategy):

    def init(self):
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(X_train, Y_train)
        self.state = 'B'

    def next(self):
        ret = self.data.ret[-1]
        forecast = self.model.predict([[self.data['Close'][-1], self.data['etheur_close'][-1], self.data['ema9'][-1],
                                        self.data['volatility'][-1]]])
        # print(ret, forecast)
        if not self.position.is_long and ret != 0 and forecast > 0.004:
            print(ret, forecast)
            # self.state = 'S'
            self.buy()
        elif not self.position.is_short and ret != 0 and forecast < 0.004:
            print(ret, forecast)
            # self.state = 'B'
            self.sell()


def statistics():
    bt = Backtest(backtest_data, Seagull, cash=100000, commission=0.004, exclusive_orders=True)
    output = bt.run()
    print(output)
    winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt():
    bt = Backtest(backtest_data, Seagull, cash=100000, commission=0.004, exclusive_orders=True)
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
