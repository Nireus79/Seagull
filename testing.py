from backtesting import Backtest, Strategy
# from meta import backtest_data, ModelSell, ModelBuy, PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy
import pandas as pd
import warnings
from sklearn.preprocessing import normalize
from data_forming import minRet, data
from synth_meta import PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy, ModelRisk

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Prado26120(Strategy):

    def init(self):
        self.MR = ModelRisk
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.cond = 'B'
        self.stop = 0
        self.profit = 0

    def next(self):
        event = self.data['event'][-1]
        TrD3 = self.data['TrD3'][-1]
        TrD6 = self.data['TrD6'][-1]
        TrD13 = self.data['TrD13'][-1]
        bbc = self.data['bb_cross'][-1]
        MAV = self.data['MAV'][-1]
        vol = self.data['Volatility'][-1]
        mom10 = self.data['mom10'][-1]
        DV = self.data['DVol'][-1]
        roc30 = self.data['roc30'][-1] / 100
        if self.cond == 'B' and event != 0 and bbc != 0 and MAV > minRet:
            features = normalize([[TrD3, TrD6, vol]])
            a, b, c = features[0][0], features[0][1], features[0][2]
            primaryPB = self.PMB.predict([[a, b, c, bbc]])[-1]
            metaPB = self.MMB.predict([[a, b, c, bbc, primaryPB]])[-1]
            ret = self.MR.predict([[a, b, c, bbc]])[-1]
            if primaryPB == metaPB and ret > minRet:
                #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                self.profit = self.data.Close * (1 + (ret + DV))
                self.stop = self.data.Close * (1 - (ret + DV))
                print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                print('{} SET + P {} S {} R {}'
                      .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret + DV))
                self.cond = 'S'
                self.buy()
        elif self.cond == 'S':
            if self.data.Close[-1] < self.stop:
                self.stop = 0
                self.profit = 0
                print('{} Sell. price {} eq {}'.
                      format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.cond = 'B'
                self.sell()
            elif self.data.Close[-1] > self.profit:
                if event != 0 and bbc != 0:
                    features = normalize([[TrD6, TrD13, mom10]])
                    a, b, c = features[0][0], features[0][1], features[0][2]
                    primaryPS = self.PMS.predict([[a, b, c, bbc]])[-1]
                    metaPS = self.MMS.predict([[a, b, c, bbc, primaryPS]])[-1]
                    featuresB = normalize([[TrD3, TrD6, vol]])
                    aB, bB, cB = featuresB[0][0], featuresB[0][1], featuresB[0][2]
                    ret = self.MR.predict([[aB, bB, cB, bbc]])[-1]
                    if primaryPS != metaPS:
                        self.stop = 0
                        self.profit = 0
                        print('{} Sell. price {} eq {}'
                              .format(self.data.index[-1], self.data.Close[-1], self.equity))
                        self.cond = 'B'
                        self.sell()
                    else:
                        if ret > minRet:
                            self.profit = self.data.Close * (1 + (ret + DV))
                            self.stop = self.data.Close * (1 - (ret + DV))
                            print('{} RESET + P {} S {} R {}'
                                  .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret + DV))


class Prado26121(Strategy):

    def init(self):
        self.MR = ModelRisk
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        # self.cond = 'B'
        self.stop = 0
        self.profit = 0

    def next(self):
        event = self.data['event'][-1]
        TrD3 = self.data['TrD3'][-1]
        TrD6 = self.data['TrD6'][-1]
        TrD13 = self.data['TrD13'][-1]
        bbc = self.data['bb_cross'][-1]
        MAV = self.data['MAV'][-1]
        Tr20 = self.data['Tr20'][-1]
        roc30 = self.data['roc30'][-1] / 100
        if not self.position and event != 0 and bbc != 0 and MAV > minRet:
            features = normalize([[TrD13, TrD6, TrD3, MAV]])
            a, b, c, d = features[0][0], features[0][1], features[0][2], features[0][3]
            primaryPB = self.PMB.predict([[a, b, c, d, bbc]])[-1]
            metaPB = self.MMB.predict([[a, b, c, d, bbc, primaryPB]])[-1]
            ret = self.MR.predict([[a, b, c, d, bbc]])[-1]
            if primaryPB == metaPB and ret > minRet and roc30 > 0:
                #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                self.profit = self.data.Close * (1 + (ret + roc30)*2)
                self.stop = self.data.Close * (1 - (ret + roc30))
                print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                print('{} SET + P {} S {} R {}'
                      .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret + roc30))
                # self.cond = 'S'
                self.buy()
        elif self.position.is_long:
            if self.data.Close[-1] < self.stop:
                self.stop = 0
                self.profit = 0
                print('{} Sell. price {} eq {}'.
                      format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.cond = 'B'
                self.position.close()
            elif self.data.Close[-1] > self.profit:
                if event != 0 and bbc != 0:
                    features = normalize([[TrD13, TrD6, TrD3, Tr20]])
                    a, b, c, d = features[0][0], features[0][1], features[0][2], features[0][3]
                    primaryPS = self.PMS.predict([[a, b, c, d, bbc]])[-1]
                    metaPS = self.MMS.predict([[a, b, c, d, bbc, primaryPS]])[-1]
                    featuresB = normalize([[TrD13, TrD6, TrD3, MAV]])
                    aB, bB, cB, dB = featuresB[0][0], featuresB[0][1], featuresB[0][2], featuresB[0][3]
                    ret = self.MR.predict([[aB, bB, cB, dB, bbc]])[-1]
                    if primaryPS != metaPS:
                        self.stop = 0
                        self.profit = 0
                        print('{} Sell. price {} eq {}'
                              .format(self.data.index[-1], self.data.Close[-1], self.equity))
                        self.cond = 'B'
                        self.position.close()
                    else:
                        if ret > minRet and roc30 > 0:
                            self.profit = self.data.Close * (1 + (ret + roc30)*2)
                            self.stop = self.data.Close * (1 - (ret + roc30))
                            print('{} RESET + P {} S {} R {}'
                                  .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret + roc30))


class Prado26240(Strategy):

    def init(self):
        self.MR = ModelRisk
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.cond = 'B'
        self.stop = 0
        self.profit = 0

    def next(self):
        event = self.data['event'][-1]
        TrD3 = self.data['TrD3'][-1]
        TrD9 = self.data['TrD9'][-1]
        TrD20 = self.data['TrD20'][-1]
        TrD6 = self.data['TrD6'][-1]
        bbc = self.data['bb_cross'][-1]
        MAV = self.data['MAV'][-1]
        roc30 = self.data['roc30'][-1] / 100
        mac = self.data['macd'][-1]
        vol = self.data['Volatility'][-1]
        diff = self.data['diff'][-1]

        if self.cond == 'B' and event != 0 and bbc != 0 and MAV > minRet:
            featuresB = normalize([[TrD9, TrD6, TrD3, diff]])
            a, b, c, d = featuresB[0][0], featuresB[0][1], featuresB[0][2], featuresB[0][3]
            primaryPB = self.PMB.predict([[a, b, c, d, bbc]])[-1]
            metaPB = self.MMB.predict([[a, b, c, d, bbc, primaryPB]])[-1]
            ret = self.MR.predict([[a, b, c, d, bbc]])[-1]
            if primaryPB == metaPB and ret > minRet and roc30 > 0:
                #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                self.profit = self.data.Close * (1 + (ret + roc30))
                self.stop = self.data.Close * (1 - (ret + roc30))
                print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                print('{} SET + P {} S {} R {} roc {}'
                      .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30))
                self.cond = 'S'
                self.buy()
        elif self.cond == 'S':
            if self.data.Close[-1] < self.stop:
                self.stop = 0
                self.profit = 0
                print('{} Sell. price {} eq {}'.
                      format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.cond = 'B'
                self.sell()
            elif self.data.Close[-1] > self.profit:
                if event != 0 and bbc != 0:
                    featuresS = normalize([[TrD20, TrD9, mac, vol]])
                    a, b, c, d = featuresS[0][0], featuresS[0][1], featuresS[0][2], featuresS[0][3]
                    primaryPS = self.PMS.predict([[a, b, c, d, bbc]])[-1]
                    metaPS = self.MMS.predict([[a, b, c, d, bbc, primaryPS]])[-1]
                    featuresB = normalize([[TrD9, TrD6, TrD3, diff]])
                    aB, bB, cB, dB = featuresB[0][0], featuresB[0][1], featuresB[0][2], featuresB[0][3]
                    ret = self.MR.predict([[aB, bB, cB, dB, bbc]])[-1]
                    if primaryPS != metaPS:
                        self.stop = 0
                        self.profit = 0
                        print('{} Sell. price {} eq {}'
                              .format(self.data.index[-1], self.data.Close[-1], self.equity))
                        self.cond = 'B'
                        self.sell()
                    else:
                        if ret > minRet and roc30 > 0:
                            self.profit = self.data.Close * (1 + (ret + roc30))
                            self.stop = self.data.Close * (1 - (ret + roc30))
                            print('{} RESET + P {} S {} R {} roc {}'
                                  .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30))


class Prado26241(Strategy):

    def init(self):
        self.MR = ModelRisk
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        # self.cond = 'B'
        self.stop = 0
        self.profit = 0

    def next(self):
        event = self.data['event'][-1]
        TrD3 = self.data['TrD3'][-1]
        Tr6 = self.data['Tr6'][-1]
        TrD6 = self.data['TrD6'][-1]
        bbc = self.data['bb_cross'][-1]
        MAV = self.data['MAV'][-1]
        roc30 = self.data['roc30'][-1] / 100
        mac4 = self.data['4Hmacd'][-1]
        vol = self.data['Volatility'][-1]
        St4H = self.data['St4H'][-1]
        momi = self.data['momi'][-1]
        rsi = self.data['rsi'][-1]

        if not self.position:
            if event != 0 and bbc != 0 and MAV > minRet:
                featuresB = normalize([[TrD3, mac4, momi, rsi]])
                a, b, c, d = featuresB[0][0], featuresB[0][1], featuresB[0][2], featuresB[0][3]
                primaryPB = self.PMB.predict([[a, b, c, d, bbc]])[-1]
                metaPB = self.MMB.predict([[a, b, c, d, bbc, primaryPB]])[-1]
                ret = self.MR.predict([[a, b, c, d, bbc]])[-1]
                if primaryPB == metaPB and ret > minRet and roc30 > 0:
                    #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                    self.profit = self.data.Close * (1 + (ret + roc30))
                    self.stop = self.data.Close * (1 - (ret + roc30))
                    print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                    print('{} SET + P {} S {} R {} roc {}'
                          .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30))
                    # self.cond = 'S'
                    self.buy()
        if self.position:
            if self.data.Close[-1] < self.stop:
                self.stop = 0
                self.profit = 0
                print('{} Sell. price {} eq {}'.
                      format(self.data.index[-1], self.data.Close[-1], self.equity))
                # self.cond = 'B'
                self.position.close()
            elif self.data.Close[-1] > self.profit:
                if event != 0 and bbc != 0:
                    featuresS = normalize([[TrD6, St4H, Tr6, vol]])
                    a, b, c, d = featuresS[0][0], featuresS[0][1], featuresS[0][2], featuresS[0][3]
                    primaryPS = self.PMS.predict([[a, b, c, d, bbc]])[-1]
                    metaPS = self.MMS.predict([[a, b, c, d, bbc, primaryPS]])[-1]
                    featuresB = normalize([[TrD3, mac4, momi, rsi]])
                    aB, bB, cB, dB = featuresB[0][0], featuresB[0][1], featuresB[0][2], featuresB[0][3]
                    ret = self.MR.predict([[aB, bB, cB, dB, bbc]])[-1]
                    if primaryPS != metaPS:
                        self.stop = 0
                        self.profit = 0
                        print('{} Sell. price {} eq {}'
                              .format(self.data.index[-1], self.data.Close[-1], self.equity))
                        # self.cond = 'B'
                        self.position.close()
                    else:
                        if ret > minRet and roc30 > 0:
                            self.profit = self.data.Close * (1 + (ret + roc30))
                            self.stop = self.data.Close * (1 - (ret + roc30))
                            print('{} RESET + P {} S {} R {} roc {}'
                                  .format(self.data.index[-1], self.profit[-1], self.stop[-1], ret, roc30))


def statistics(dt, strategy):
    bt = Backtest(dt, strategy, cash=100000, commission=0.026, exclusive_orders=True)
    output = bt.run()
    print(output)
    # winsound.Beep(1000, 1500)
    bt.plot(resample=False)


def opt(dt, strategy):
    bt = Backtest(dt, strategy, cash=100000, commission=0.026, exclusive_orders=True)
    stats, heatmap = bt.optimize(
        t=range(-200, 201),
        # sl=range(1, 6),
        maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        # maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    # winsound.Beep(1000, 1500)


test_data = data
statistics(test_data, Prado26121)
# opt(test_data, Prelder)
