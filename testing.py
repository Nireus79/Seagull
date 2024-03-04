import winsound
from backtesting import Backtest, Strategy
from meta import backtest_data, ModelSell, ModelBuy, PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy
import pandas as pd
import warnings
from sklearn.preprocessing import normalize

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# PrimeModelBuy = joblib.load('PrimeModelBuy.pkl')
# MetaModelBuy = joblib.load('MetaModelBuy.pkl')
# PrimeModelSell = joblib.load('PrimeModelSell.pkl')
# MetaModelSell = joblib.load('MetaModelSell.pkl')


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
    mav = 10
    mavs = -100

    def init(self):
        self.CMB = ModelBuy
        self.CMS = ModelSell
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.cond = 'B'
        self.stop = 0

    def next(self):
        event = self.data['event'][-1]
        TrD3 = self.data['TrD3'][-1]
        TrD6 = self.data['TrD6'][-1]
        TrD13 = self.data['TrD13'][-1]
        bbc = self.data['bb_cross'][-1]
        bb_sq = self.data['bb_sq'][-1]
        St4 = self.data['St4H'][-1]
        DS4 = self.data['4H%DS'][-1]
        mac = self.data['macd'][-1]
        vdiff = self.data['vdiff'][-1]
        vol = self.data['Volatility'][-1]

        atr = self.data['atr'][-1]
        rsi = self.data['rsi'][-1]
        MAV = self.data['MAV'][-1]
        MAVS = self.data['MAV_signal'][-1]
        if self.cond == 'B' and event != 0 and bbc != 0 and MAV > 0.026:
            features = [[TrD6, mac, vdiff]]
            features = normalize(features)
            a, b, c = features[0][0], features[0][1], features[0][2]
            # classicPB = self.CMB.predict([[a, b, bbc]])
            primaryPB = self.PMB.predict([[a, b, c, bbc]])[-1]
            metaPB = self.MMB.predict([[a, b, c, bbc, primaryPB]])[-1]
            print(self.data.index[-1], primaryPB, metaPB)
            if primaryPB == 1.0 and metaPB == 1:
                self.stop = self.data.Close[-2] - atr
                print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.cond = 'S'
                self.buy()
        # elif self.cond == 'S' and self.data.Close[-1] < self.stop:
        #     self.cond = 'B'
        #     self.stop = 0
        #     print('{} P Sell. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
        #     self.sell()
        elif self.cond == 'S' and event != 0 and bbc != 0:
            features = [[TrD13, TrD3, mac]]
            features = normalize(features)
            a, b, c = features[0][0], features[0][1], features[0][2]
            # classicPS = self.CMS.predict([[a, b, bbc]])
            primaryPS = self.PMS.predict([[a, b, c, bbc]])[-1]
            metaPS = self.MMS.predict([[a, b, c, bbc, primaryPS]])[-1]
            print(self.data.index[-1], primaryPS, metaPS)
            if primaryPS == 1.0 and metaPS == 1:
                self.stop = 0
                print('{} Sell. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.cond = 'B'
                self.sell()


def statistics(data, strategy):
    bt = Backtest(data, strategy, cash=100000, commission=0.026, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt(data, strategy):
    bt = Backtest(data, strategy, cash=100000, commission=0.026, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        # ema=range(5, 31, 1),
        # rs=range(50, 76, 1),
        mav=range(-100, 100, 1),
        # mavs=range(0, 100, 1),
        maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        # maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    # winsound.Beep(1000, 1500)


statistics(backtest_data, Prelder)

# opt(backtest_data, Prelder)
