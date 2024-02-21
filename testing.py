import winsound
from backtesting import Backtest, Strategy
from meta import backtest_data, ModelSell, ModelBuy, PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# class SIM(Strategy):
#
#     def init(self):
#         self.cond = 'B'
#
#     def next(self):
#         B = self.data['Buy'][-1]
#         S = self.data['Sell'][-1]
#         sim_data['NAV'].loc[self.data.index[-1]] = self.equity / 100000
#         if self.cond == 'B' and B:
#             self.cond = 'S'
#             self.buy()
#         elif self.cond == 'S' and S:
#             self.cond = 'B'
#             self.sell()


class Prelder(Strategy):

    def init(self):
        self.CMB = ModelBuy
        self.CMS = ModelSell
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
        vol = self.data['Volatility'][-1]

        if self.cond == 'B' and ret != 0 and bbc != 0 and vol > 0.01:
            classicPB = self.CMB.predict([[vol, st4, tr]])[-1]
            # primaryPB = self.PMB.predict([[vol, st4, tr]])[-1]
            # metaPB = self.MMB.predict([[vol, st4, tr, primaryPB]])[-1]
            if classicPB == 1:
                self.buy_price = self.data.Close[-1]
                # print(self.data.index[-1], 'Buy at: ', self.buy_price)
                backtest_data['Buy'].loc[self.data.index[-1]] = True
                self.cond = 'S'
                self.buy()
        elif self.cond == 'S' and ret != 0 and vol > 0.01:
            classicPS = self.CMS.predict([[bbc, bbl, tr]])[-1]
            # primaryPS = self.PMS.predict([[bbc, bbl, tr]])[-1]
            # metaPS = self.MMS.predict([[bbc, bbl, tr, primaryPS]])[-1]
            if classicPS == 0:
                self.sell_price = self.data.Close[-1]
                self.sell_price = close
                # print(self.data.index[-1], 'Model sell at:', self.sell_price, 'Profit: ',
                #       self.sell_price - self.buy_price, 'Balance:', self.equity)
                self.sell_price = self.buy_price = 0
                backtest_data['Sell'].loc[self.data.index[-1]] = True
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
    bt = Backtest(data, strategy, cash=100000, commission=0.0026, exclusive_orders=True)
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


backtest_data['Buy'] = False
backtest_data['Sell'] = False

statistics(backtest_data, Prelder)
# # print(backtest_data.columns)
# print(backtest_data)
# backtest_data.to_csv('bt_data3.csv')
# opt(backtest_data, Prelder)

# print(backtest_data)
# print(backtest_data.b.sum())
# print(backtest_data.s.sum())
# opt()

# sim_data = pd.read_csv('bt_data249322.csv')
# sim_data['NAV'] = 1.0
# sim_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
# sim_data['Date'] = pd.to_datetime(sim_data['Date'])
# # sim_data.set_index('Date', inplace=True)
# statistics(sim_data, SIM)
# print(sim_data)
# sim_data.to_csv('NV.csv')
