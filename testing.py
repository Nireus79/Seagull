import winsound
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from data_forming import X, Y, X_train, X_test, Y_train, Y_test, full_data, backtest_data
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.metrics import classification_report, mean_squared_error
from toolbox import create_LSTMmodel, standardizer
# from meta import model1, model2, meta_backtest_data
import warnings

warnings.filterwarnings('ignore')


class Prelder(Strategy):

    def init(self):
        self.model = MLPClassifier(
            activation='relu',
            alpha=0.0001,
            hidden_layer_sizes=(100,),
            learning_rate='constant',
            solver='adam'
        )
        self.model.fit(X_train, Y_train)
        self.cond = 'B'
        self.buy_price = 0
        self.sell_price = 0

    def next(self):
        ret = self.data['ret'][-1]
        close = self.data.Close[-1]
        Dema9 = self.data['Dema9'][-1]
        K = self.data['4H%K'][-1]
        vol = self.data['Volatility'][-1]
        # atr = self.data['4H_atr'][-1]
        forecast = self.model.predict([[close, Dema9, K, vol]])

        if self.cond == 'B':
            if ret != 0 and forecast == 1:
                self.buy_price = self.data.Close[-1]
                print(self.data.index[-1], 'Buy at: ', self.buy_price)
                self.cond = 'S'
                full_data['b'].loc[self.data.index[-1]] = True
                self.buy()
        elif self.cond == 'S':
            if ret != 0 and forecast == 0:
                self.sell_price = self.data.Close[-1]
                print(self.data.index[-1], 'Sell at:', self.sell_price, 'Profit: ', self.sell_price - self.buy_price)
                self.sell_price = self.buy_price = 0
                self.cond = 'B'
                full_data['s'].loc[self.data.index[-1]] = True
                self.sell()


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
statistics(backtest_data, Prelder)
# opt(backtest_data, Prelder)

# print(backtest_data)
# print(backtest_data.b.sum())
# print(backtest_data.s.sum())
# opt()
