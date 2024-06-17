from backtesting import Backtest, Strategy
# from meta import backtest_data, ModelSell, ModelBuy, PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy
import pandas as pd
import warnings
from sklearn.preprocessing import normalize
from data_forming import minRet, data
# from synth_meta import PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy, ModelRisk, test_data
import numpy as np
import joblib
warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

PrimeModelBuy = joblib.load('csv/pkl/PrimeModelBuy.pkl')
MetaModelBuy = joblib.load('csv/pkl/MetaModelBuy.pkl')
PrimeModelSell = joblib.load('csv/pkl/PrimeModelSell.pkl')
MetaModelSell = joblib.load('csv/pkl/MetaModelSell.pkl')
ModelRisk = joblib.load('csv/pkl/ModelRisk.pkl')
# data = pd.read_csv('csv/tb/eth001.csv')
# # data.set_index('time', inplace=True)
# minRet = 0.01
fee = 0.04

# test_data = test_data[:100]
# test_data['Open'] = 1
# test_data['High'] = 1
# test_data['Low'] = 1
# test_data['event'] = 1



class Prelder(Strategy):
    RetMin = 0.03
    pt = 2
    sl = 1

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
        close = self.data['Close'][-1]

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
                if primaryPB == metaPB and ret > self.RetMin and roc30 > 0:
                    #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                    self.profit = self.data.Close * (1 + ((ret + roc30r) * self.pt))
                    self.stop = self.data.Close * (1 - ((ret + roc30r) * self.sl))
                    # self.timestamp = self.data.t[-1]
                    print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                    print('{} SET + P {} S {} R {} roc30 {}'
                          .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30))
                    self.buy()
        elif self.position:
            if close < self.stop:
                self.stop = 0
                self.profit = 0
                print('{} - Sell Stop. price {} eq {}'.
                      format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.position.close()
            elif close > self.profit:  # or self.data.t[-1] > self.timestamp + 86400000:
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
                        print('{} - Sell Profit. price {} eq {}'
                              .format(self.data.index[-1], self.data.Close[-1], self.equity))
                        self.position.close()
                    else:
                        if ret > self.RetMin and roc30 > 0:
                            self.profit = self.data.Close * (1 + ((ret + roc30r) * self.pt))
                            self.stop = self.data.Close * (1 - ((ret + roc30r) * self.sl))
                            #self.timestamp = self.data.t[-1]
                            print('{} RESET + P {} S {} R {} roc30 {}'
                                  .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30))


class PrelderLimit(Strategy):

    def init(self):
        self.MR = ModelRisk
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.stop = 0
        self.profit = 0
        # self.buy_lim = 0
        # self.sell_lim = 0
        self.timestamp = 0

    def next(self):
        event = self.data['event'][-1]
        TrD20 = self.data['TrD20'][-1]
        TrD3 = self.data['TrD3'][-1]
        D4 = self.data['4H%D'][-1]
        Tr6 = self.data['Tr6'][-1]
        bbc = self.data['bb_cross'][-1]
        roc30 = self.data['roc30'][-1]
        mac4 = self.data['4Hmacd'][-1]
        vol = self.data['Volatility'][-1]
        vv = self.data['VV'][-1]
        rsi = self.data['rsi'][-1]
        bb_l = self.data['bb_l'][-1]
        high4 = self.data['4H_High'][-2]
        low4 = self.data['4H_Low'][-2]
        atr4 = self.data['4H_atr'][-1]
        close = self.data['Close'][-1]
        index = self.data.index[-1]
        if not self.position:
            if self.profit != 0:
                if close > self.profit:
                    print('{} Price {} > Buy lim {}.'.format(index, close, self.profit))
                    self.profit = self.stop = 0
                    print('{} + Buy at {} eq {}'.format(index, close, self.equity))
                    self.timestamp = self.data.t[-1]
                    self.buy()
                elif close < self.stop: # or self.data.t[-1] > self.timestamp + 86400000:
                    print('{} Price {} < Stop {}.'.format(index, close, self.stop))
                    self.profit = self.stop = self.timestamp = 0
            else:
                if event > minRet and bbc != 0:
                    featuresB = [[TrD20, TrD3, mac4, vol, vv, roc30, rsi]]
                    featuresB = normalize(featuresB)
                    featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
                    primaryPB = self.PMB.predict([featuresB])[-1]
                    featuresMB = featuresB
                    featuresMB = np.insert(featuresMB, len(featuresMB), primaryPB)
                    metaPB = self.MMB.predict([featuresMB])[-1]
                    ret = self.MR.predict([featuresB])[-1]
                    if primaryPB == metaPB and ret > fee and roc30 > 0:
                        print('{} PPB {} MPB {} ret {} roc {}'.format(index, primaryPB, metaPB, ret, roc30))
                        self.profit = low4
                        self.stop = high4
                        print('Setting Profit {} Stop {}'.format(self.profit, self.stop))
        elif self.position:
            if self.stop != 0:
                if close < self.stop:
                    self.profit = self.stop = self.timestamp = 0
                    print('{} - Sell Stop at {} eq {}'.format(index, close, self.equity))
                    self.position.close()
                elif close > self.profit:
                    if event != 0 and bbc != 0:
                        featuresB = [[TrD20, TrD3, mac4, vol, vv, roc30, rsi]]
                        featuresB = normalize(featuresB)
                        featuresB = np.insert(featuresB, len(featuresB[0]), bbc)
                        primaryPB = self.PMB.predict([featuresB])[-1]
                        featuresMB = featuresB
                        featuresMB = np.insert(featuresMB, len(featuresMB), primaryPB)
                        metaPB = self.MMB.predict([featuresMB])[-1]
                        ret = self.MR.predict([featuresB])[-1]
                        if primaryPB == metaPB and ret > fee and roc30 > 0:
                            print('{} PPB {} MPB {} ret {} roc {}'.format(index, primaryPB, metaPB, ret, roc30))
                            self.profit = low4
                            self.stop = high4
                            print('Resetting Profit {} Stop {}'.format(self.profit, self.stop))
            else:
                if event != 0 and bbc != 0:
                    featuresS = [[TrD20, TrD3, D4, mac4, Tr6, roc30, bb_l, rsi]]
                    featuresS = normalize(featuresS)
                    featuresS = np.insert(featuresS, len(featuresS[0]), bbc)
                    primaryPS = self.PMS.predict([featuresS])[-1]
                    featuresMS = featuresS
                    featuresMS = np.insert(featuresMS, len(featuresMS), primaryPS)
                    metaPS = self.MMS.predict([featuresMS])[-1]
                    if primaryPS != metaPS:
                        print('{} PPS {} MPS {}'.format(index, primaryPS, metaPS))
                        self.profit = low4
                        self.stop = high4
                        print('{} Setting Limit {} Stop {}'.format(index, self.profit, self.stop))




def statistics(dt, strategy):
    # EON commission=0.045
    bt = Backtest(dt, strategy, cash=100000, commission=fee, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt(dt, strategy):
    # EON commission=0.045
    bt = Backtest(dt, strategy, cash=100000, commission=fee, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        # pt=range(1, 21),
        # sl=range(1, 21),
        RetMin=range(0, 21),
        # maximize='Sharpe Ratio',
        maximize='Equity Final [$]',
        # maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    # winsound.Beep(1000, 1500)



statistics(data, Prelder)
# opt(data, Prelder)
