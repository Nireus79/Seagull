from backtesting import Backtest, Strategy
from meta import backtest_data, ModelSell, ModelBuy, PrimeModelSell, PrimeModelBuy, MetaModelSell, MetaModelBuy
import pandas as pd
import warnings
from sklearn.preprocessing import normalize
from data_forming import minRet

warnings.filterwarnings('ignore')
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Frequency(Strategy):
    ret = minRet

    def init(self):
        self.CMB = ModelBuy
        self.CMS = ModelSell
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.cond = 'B'
        self.stop = 0
        self.profit = 0
        self.timestamp_reset = self.timestamp = self.data.index[-1]

    def next(self):
        event = self.data['event'][-1]
        Tr6 = self.data['Tr6'][-1]
        TrD3 = self.data['TrD3'][-1]
        TrD6 = self.data['TrD6'][-1]
        st4 = self.data['St4H'][-1]
        mac4 = self.data['4Hmacd'][-1]
        bbc = self.data['bb_cross'][-1]
        MAV = self.data['MAV'][-1]
        vol = self.data['Volatility'][-1]
        mom20 = self.data['mom20'][-1]
        mac = self.data['macd'][-1]
        bbl = self.data['bb_l'][-1]
        if self.cond == 'B' and event != 0 and bbc != 0 and MAV > self.ret:
            #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
            # print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
            # print('{} SET P {} S {}'.format(self.data.index[-1], self.profit, self.stop))
            self.cond = 'S'
            self.buy()
        elif self.cond == 'S':
            if event != 0 and bbc != 0 and MAV > self.ret:
                # print(self.data.index[-1], primaryPS, metaPS)
                # print('{} Sell. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                self.cond = 'B'
                self.sell()


# 10 1030  870    207717.78678
# 9 1120  900    132053.28794
# 8 1170  840    87400.59820
# 7 1730  980    126289.84188
# 6 1240  830    168557.95302
# 5 2990  990    136403.14558
# 4 1140  800    126967.78086
# 3 1200  930    359933.1271
# 2 1130  820    257896.67188
# 1 1310  830    115968.59634
# med 1220 885
#     1406 880
class Prelder(Strategy):
    pt = 1020
    sl = 975
    t = 14
    c = 140

    def init(self):
        self.CMB = ModelBuy
        self.CMS = ModelSell
        self.PMB = PrimeModelBuy
        self.PMS = PrimeModelSell
        self.MMB = MetaModelBuy
        self.MMS = MetaModelSell
        self.cond = 'B'
        self.stop = 0
        self.profit = 0
        self.timestamp_reset = self.timestamp = self.data.index[-1]

    def next(self):
        event = self.data['event'][-1]
        Tr6 = self.data['Tr6'][-1]
        TrD3 = self.data['TrD3'][-1]
        TrD6 = self.data['TrD6'][-1]
        st4 = self.data['St4H'][-1]
        mac4 = self.data['4Hmacd'][-1]
        bbc = self.data['bb_cross'][-1]
        MAV = self.data['MAV'][-1]
        vol = self.data['Volatility'][-1]
        mom20 = self.data['mom20'][-1]
        mac = self.data['macd'][-1]
        bbl = self.data['bb_l'][-1]
        if self.cond == 'B' and event != 0 and bbc != 0 and MAV > 0.026:
            features = [[TrD3, mac4, mom20, TrD6, bbl]]
            features = normalize(features)
            a, b, c, d, e = features[0][0], features[0][1], features[0][2], features[0][3], features[0][4]
            # classicPB = self.CMB.predict([[a, b, c, d, bbc]])
            primaryPB = self.PMB.predict([[a, b, c, d, e]])[-1]
            metaPB = self.MMB.predict([[a, b, c, d, e, primaryPB]])[-1]
            risk = self.CMB.predict([[a, b, c, d, e]])[-1]
            # print(self.data.index[-1], primaryPB, metaPB)
            if primaryPB == metaPB:
                #  𝜋− =−.01,𝜋+ = .005 are set by the portfolio manager
                self.profit = self.data.Close * (1 + risk)
                self.stop = self.data.Close * ((1 - risk) / 2)
                # self.timestamp = self.data.index[-1] + pd.Timedelta(hours=self.t)
                # print('{} Buy. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                # print('{} SET P {} S {}'.format(self.data.index[-1], self.profit, self.stop))
                self.cond = 'S'
                self.buy()
        elif self.cond == 'S':
            if self.data.Close[-1] > self.profit or \
                    self.data.Close[-1] < self.stop:
                if event != 0 and bbc != 0:
                    features = [[TrD6, st4, mom20, mac, MAV]]
                    features = normalize(features)
                    a, b, c, d, e = features[0][0], features[0][1], features[0][2], features[0][3], features[0][4]
                    # classicPS = self.CMS.predict([[a, b, c, d, bbc]])
                    primaryPS = self.PMS.predict([[a, b, c, d, e]])[-1]
                    metaPS = self.MMS.predict([[a, b, c, d, e, primaryPS]])[-1]
                    risk = self.CMB.predict([[a, b, c, d, e]])[-1]
                    # print(self.data.index[-1], primaryPS, metaPS)
                    if primaryPS != metaPS:
                        self.stop = 0
                        self.profit = 0
                        self.timestamp = self.timestamp_reset
                        # print('{} Sell. price {} eq {}'.format(self.data.index[-1], self.data.Close[-1], self.equity))
                        self.cond = 'B'
                        self.sell()
                    else:
                        self.profit = self.data.Close * (1 + risk)
                        self.stop = self.data.Close * ((1 - risk) / 2)
                        # self.timestamp = self.data.index[-1]
                        # print('{} RESET P {} S {}'.format(self.data.index[-1], self.profit, self.stop))


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
        c=range(0, 1000, 10),
        # pt=range(1000, 3000, 10),
        # sl=range(800, 1000, 10),
        maximize='Sharpe Ratio',
        # maximize='Equity Final [$]',
        # maximize='Max. Drawdown [%]',
        return_heatmap=True)
    print(stats)
    print(heatmap.sort_values().iloc[-100:])
    # winsound.Beep(1000, 1500)


statistics(backtest_data, Prelder)

# opt(backtest_data, Prelder)

# Volatility   -0.139562
# event        -0.139562
# MAV          -0.133706
# vema13       -0.115857
# vema9        -0.109713
# vema20       -0.108361
# VtrD3        -0.099991
# bb_l         -0.093706
# vmom20       -0.075175
# vema6        -0.074727
# bb_sq        -0.073293
# vmom30       -0.069781
# Volume       -0.067521
# Vtr3         -0.065105
# VtrD6        -0.062414
# vema3        -0.059450
# Vol_Vol      -0.052043
# Vtr6         -0.049039
# VtrD9        -0.045171
# atr          -0.042500
# srl_corr     -0.031338
# Vtr20        -0.029010
# bb_t         -0.027315
# VtrD13       -0.027184
# 4Hmacd       -0.026815
# vmom10       -0.025634
# Vtr13        -0.021442
# 4H_atr       -0.020782
# vcusum       -0.020630
# Vtr9         -0.020245
# St4H         -0.013724
# cusum        -0.013448
# Dema3        -0.005921
# VtrD20       -0.004368
# vdiff         0.001104
# Dvema20       0.001122
# Dema9         0.001656
# Dema13        0.003327
# ema20         0.003678
# ema13         0.004345
# Dema20        0.004696
# ema9          0.005063
# ema6          0.005411
# 4H_Low        0.005737
# ema3          0.006986
# 4H_High       0.007297
# Close         0.008704
# vsrl_corr     0.014160
# Vsrl_corr     0.015658
# vroc30        0.016001
# MAV_signal    0.022237
# Dvema13       0.023177
# vroc10        0.024152
# macd          0.033143
# StD           0.039948
# 4H%K          0.040393
# Dvema9        0.040584
# vrsi          0.041761
# 4H%D          0.043827
# 4H%DS         0.052948
# vmacd         0.056032
# Dvema6        0.057279
# Tr9           0.066734
# Tr6           0.069811
# Tr13          0.072444
# TrD20         0.073663
# Tr20          0.073807
# mom10         0.074007
# bb_cross      0.075851
# 4H_rsi        0.079537
# %K            0.085163
# diff          0.086200
# rsi           0.086654
# %DS           0.089031
# mom20         0.089582
# TrD13         0.091795
# Dvema3        0.093771
# %D            0.094613
# TrD9          0.115680
# mom30         0.116717
# roc20         0.117500
# roc10         0.125849
# vroc20        0.143582
# TrD6          0.144281
# roc30         0.146058
# TrD3          0.209740
# bin           0.696385
# ret           1.000000
