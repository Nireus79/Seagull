from backtesting import Backtest, Strategy
# from meta import backtest_data, ModelSell, ModelBuy, PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy
import pandas as pd
import warnings
from sklearn.preprocessing import normalize
from data_forming import minRet, data
from synth_meta import PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy, ModelRisk, test_data
import numpy as np
warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# test_data = test_data[:100]
# test_data['Open'] = 1
# test_data['High'] = 1
# test_data['Low'] = 1
# test_data['event'] = 1



class Prelder(Strategy):
    pt = 1
    sl = 0.5

    def init(self):
        self.MR = ModelRisk
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.stop = 0
        self.profit = 0
        # self.timestamp = 0

    def next(self):
        event = self.data['event'][-1]
        TrD20 = self.data['TrD20'][-1]
        TrD3 = self.data['TrD3'][-1]
        D4 = self.data['4H%D'][-1]
        Tr6 = self.data['Tr6'][-1]
        bbc = self.data['bb_cross'][-1]
        roc30 = self.data['roc30'][-1]
        roc30r = self.data['roc30'][-1] / 100
        mac4 = self.data['4Hmacd'][-1]
        vol = self.data['Volatility'][-1]
        vv = self.data['VV'][-1]
        rsi = self.data['rsi'][-1]
        bb_l = self.data['bb_l'][-1]

        if not self.position:
            if event > minRet and bbc != 0:
                featuresB = [[TrD20, TrD3, mac4, vol, vv, roc30, rsi]]
                featuresB = normalize(featuresB)
                featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
                primaryPB = self.PMB.predict([featuresB])[-1]
                featuresMB = featuresB
                featuresMB = np.insert(featuresMB, len(featuresMB), primaryPB)
                metaPB = self.MMB.predict([featuresMB])[-1]
                ret = self.MR.predict([featuresB])[-1]
                if primaryPB == metaPB and ret > minRet and roc30 > 0:
                    #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                    self.profit = self.data.Close * (1 + ((ret + roc30r) * self.pt))
                    self.stop = self.data.Close * (1 - ((ret + roc30r) * self.sl))
                    # self.timestamp = self.data.t[-1]
                    print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                    print('{} SET + P {} S {} R {} roc {}'
                          .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30r))
                    self.buy()
        elif self.position:
            if self.data.Close[-1] < self.stop:
                self.stop = 0
                self.profit = 0
                #self.timestamp = 0
                print('{} Sell Stop. price {} eq {}'.
                      format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.position.close()
            elif self.data.Close[-1] > self.profit:  # or self.data.t[-1] > self.timestamp + 86400000:
                if event != 0 and bbc != 0:
                    featuresS = [[TrD20, TrD3, D4, mac4, Tr6, roc30, bb_l, rsi]]
                    featuresS = normalize(featuresS)
                    featuresS = np.insert(featuresS, len(featuresS[0]), bbc)
                    primaryPS = self.PMS.predict([featuresS])[-1]
                    featuresMS = featuresS
                    featuresMS = np.insert(featuresMS, len(featuresMS), primaryPS)
                    metaPS = self.MMS.predict([featuresMS])[-1]
                    featuresB = [[TrD20, TrD3, mac4, vol, vv, roc30, rsi]]
                    featuresB = normalize(featuresB)
                    featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
                    ret = self.MR.predict([featuresB])[-1]
                    if primaryPS != metaPS:
                        self.stop = 0
                        self.profit = 0
                        # self.timestamp = 0
                        print('{} Sell Profit. price {} eq {}'
                              .format(self.data.index[-1], self.data.Close[-1], self.equity))
                        self.position.close()
                    else:
                        if ret > minRet and roc30 > 0:
                            self.profit = self.data.Close * (1 + ((ret + roc30r) * self.pt))
                            self.stop = self.data.Close * (1 - ((ret + roc30r) * self.sl))
                            #self.timestamp = self.data.t[-1]
                            print('{} RESET + P {} S {} R {} roc {}'
                                  .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30r))


def statistics(dt, strategy):
    # EON commission=0.045
    bt = Backtest(dt, strategy, cash=100000, commission=0.045, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt(dt, strategy):
    # EON commission=0.045
    bt = Backtest(dt, strategy, cash=100000, commission=0.045, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        pt=range(1, 21),
        sl=range(1, 21),
        maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        # maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    # winsound.Beep(1000, 1500)


tst_data = data
statistics(tst_data, Prelder)
# opt(tst_data, Prelder)
