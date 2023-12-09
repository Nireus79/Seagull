import winsound
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from data_forming import X_trainB, X_testB, Y_trainB, Y_testB, X_trainS, X_testS, Y_trainS, Y_testS, \
    full_data, backtest_dataB, backtest_dataS
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, mean_squared_error
from toolbox import create_LSTMmodel, standardizer
import warnings

warnings.filterwarnings('ignore')

modelB = MLPClassifier()
modelB.fit(X_trainB, Y_trainB)
predB = modelB.predict(X_testB)
modelS = KNeighborsClassifier()
modelS.fit(X_trainS, Y_trainS)
predS = modelS.predict(X_testS)
print('Model B')
print(classification_report(Y_testB, predB, target_names=['no_trade', 'trade']))
print('model S')
print(classification_report(Y_testS, predS, target_names=['no_trade', 'trade']))


class Prelder(Strategy):

    def init(self):
        self.MB = modelB
        self.MS = modelS
        self.cond = 'B'
        self.buy_price = 0
        self.sell_price = 0
        self.stop = 0

    def next(self):
        ret = self.data['ret'][-1]
        close = self.data['Close'][-1]
        Dema9 = self.data['Dema9'][-1]
        K = self.data['4H%K'][-1]
        D = self.data['4H%D'][-1]
        low = self.data['4H_Low'][-2]
        atr = self.data['4H_atr'][-1]
        # forecastB = self.MB.predict([[close, Dema9, K, D]])
        # forecastS = self.MS.predict([[K, D]])
        if self.cond == 'B' and ret != 0 and self.MB.predict([[close, Dema9, K, D]]) == 1:
                self.buy_price = self.data.Close[-1]
                print(self.data.index[-1], 'Buy at: ', self.buy_price)
                full_data['b'].loc[self.data.index[-1]] = True
                self.cond = 'S'
                self.stop = low - atr
                self.buy()
        elif self.cond == 'S':
            if close < self.stop:
                self.sell_price = close
                print(self.data.index[-1], 'Pillow sell at:', self.sell_price, 'Profit: ',
                      self.sell_price - self.buy_price)
                self.sell_price = self.buy_price = 0
                full_data['s'].loc[self.data.index[-1]] = True
                self.stop = 0
                self.cond = 'B'
                self.sell()
            elif ret != 0 and self.MS.predict([[K, D]]) == 1:
                self.sell_price = self.data.Close[-1]
                self.sell_price = close
                print(self.data.index[-1], 'Model sell at:', self.sell_price, 'Profit: ',
                      self.sell_price - self.buy_price)
                self.sell_price = self.buy_price = 0
                full_data['s'].loc[self.data.index[-1]] = True
                self.cond = 'B'
                self.stop = 0
                self.sell()
            elif close < self.data['Close'][-2]:
                self.stop = low - atr


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
statistics(backtest_dataB, Prelder)
# opt(backtest_data, Prelder)

# print(backtest_data)
# print(backtest_data.b.sum())
# print(backtest_data.s.sum())
# opt()
