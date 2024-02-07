import winsound
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from data_forming import full_data
from meta import PrimeModelBuy, MetaModelBuy, PrimeModelSell, MetaModelSell, backtest_data
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
        close = self.data['Close'][-1]
        ret = self.data['ret'][-1]
        bbc = self.data['bb_cross'][-1]
        bbl = self.data['bb_l'][-1]
        tr = self.data['TrD3'][-1]
        st4 = self.data['St4H'][-1]

        if self.cond == 'B' and ret != 0 and bbc != 0:
            primaryPB = self.PMB.predict([[st4, tr]])[-1]
            metaPB = self.MMB.predict([[st4, tr, primaryPB]])[-1]
            if primaryPB == metaPB == 1:
                self.buy_price = self.data.Close[-1]
                print(self.data.index[-1], 'Buy at: ', self.buy_price)
                full_data['b'].loc[self.data.index[-1]] = True
                self.cond = 'S'
                self.buy()
        elif self.cond == 'S' and ret != 0:
            primaryPS = self.PMS.predict([[bbc, bbl, tr]])[-1]
            metaPS = self.MMS.predict([[bbc, bbl, tr, primaryPS]])[-1]
            if primaryPS == 0 and metaPS == 1:
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
