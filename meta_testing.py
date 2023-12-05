import winsound
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import classification_report, mean_squared_error
from toolbox import create_LSTMmodel, standardizer
from data_forming import full_data
from meta import modelPrime, modelMeta, meta_backtest_data
import warnings

warnings.filterwarnings('ignore')


class MetaPrelder(Strategy):

    def init(self):
        self.cond = 'B'
        self.buy_price = 0
        self.sell_price = 0

    def next(self):
        ret = self.data['ret'][-1]
        trend = self.data['trend'][-1]
        momentum = self.data['momentum'][-1]
        elder = self.data['elder'][-1]
        forecast1 = modelPrime.predict([[trend, momentum, elder]])[-1]
        forecast2 = modelMeta.predict([[trend, momentum, elder, forecast1]])

        if self.cond == 'B':
            if ret != 0 and forecast1 == forecast2 == 1:
                self.buy_price = self.data.Close[-1]
                print(self.data.index[-1], 'Buy at: ', self.buy_price)
                self.cond = 'S'
                full_data['b'].loc[self.data.index[-1]] = True
                self.buy()
        elif self.cond == 'S':
            if ret != 0 and forecast1 == forecast2 == 0:
                self.sell_price = self.data.Close[-1]
                print(self.data.index[-1], 'Sell at:', self.sell_price, 'Profit: ', self.sell_price - self.buy_price)
                self.sell_price = self.buy_price = 0
                self.cond = 'B'
                full_data['s'].loc[self.data.index[-1]] = True
                self.position.close()


def statistics(data, strategy):
    bt = Backtest(data, strategy, cash=100000, commission=0.045, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt(data, strategy):
    bt = Backtest(data, strategy, cash=100000, commission=0.045, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        # ema=range(5, 31, 1),
        # rs=range(50, 76, 1),
        lower_bound=range(10, 30, 1),
        upper_bound=range(70, 100, 1),
        # maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    winsound.Beep(1000, 1500)


full_data['b'] = 0
full_data['s'] = 0
statistics(meta_backtest_data, MetaPrelder)
# opt(backtest_data, Prelder)

# print(backtest_data)
# print(backtest_data.b.sum())
# print(backtest_data.s.sum())
# opt()
