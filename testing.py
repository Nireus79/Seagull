import winsound
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from data_forming import full_data, backtest_data
from meta import PrimeModelBuy, PrimeModelSell, MetaModelBuy, MetaModelSell
import warnings

warnings.filterwarnings('ignore')


class Prelder(Strategy):

    def init(self):
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.cond = 'B'
        self.buy_price = 0
        self.sell_price = 0

    def next(self):
        ret = self.data['ret'][-1]
        close = self.data['Close'][-1]
        Dema9 = self.data['Dema9'][-1]
        Dema13 = self.data['Dema13'][-1]
        mac = self.data['4Hmacd'][-1]
        K = self.data['4H%K'][-1]
        D = self.data['4H%D'][-1]
        # forecastB = self.MB.predict([[close, Dema9, K, D]])
        # forecastS = self.MS.predict([[K, D]])
        if self.cond == 'B' and ret != 0 and \
                self.PMB.predict([[Dema9, Dema13, mac]]) == self.MMB.predict([[Dema9, Dema13, mac]]) == 1:
            self.buy_price = self.data.Close[-1]
            print(self.data.index[-1], 'Buy at: ', self.buy_price)
            full_data['b'].loc[self.data.index[-1]] = True
            self.cond = 'S'
            self.buy()
        elif self.cond == 'S':
            if ret != 0 and self.PMS.predict([[K, D]]) == self.MMS.predict([[K, D]]) == 1:
                self.sell_price = self.data.Close[-1]
                self.sell_price = close
                print(self.data.index[-1], 'Model sell at:', self.sell_price, 'Profit: ',
                      self.sell_price - self.buy_price)
                self.sell_price = self.buy_price = 0
                full_data['s'].loc[self.data.index[-1]] = True
                self.cond = 'B'
                self.stop = 0
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
