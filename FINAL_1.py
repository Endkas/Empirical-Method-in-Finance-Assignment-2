#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 12:46:57 2021

@author: Endrit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import norm
import seaborn as sns
from statsmodels.tsa.ar_model import AutoReg
import math



#==================================================
#Downloading the data
#==================================================



Bitcoin = pd.read_csv("Part1_BTCUSD_d.csv")
Bitcoin = Bitcoin.iloc[:1613, [1, 6]]
Bitcoin = Bitcoin.rename(columns = {"date" : "Date", "close" : "BTC/USD"})
Bitcoin['Date'] = pd.to_datetime(Bitcoin['Date']).dt.date
Bitcoin = Bitcoin.set_index('Date')


Ethereum = pd.read_csv("Part1_ETHUSD_d.csv")
Ethereum = Ethereum.iloc[:1613, [1, 6]]
Ethereum = Ethereum.rename(columns = {"date" : "Date", "close" : "ETH/USD"})
Ethereum['Date'] = pd.to_datetime(Ethereum['Date']).dt.date
Ethereum = Ethereum.set_index('Date')


Monera = pd.read_csv("Part1_XMRUSD_d.csv")
Monera = Monera.iloc[1:, [1, 6]]
Monera = Monera.rename(columns = {"date" : "Date", "close" : "XMR/USD"})
Date_XMR = Monera.iloc[:, 0]
Monera['Date'] = pd.to_datetime(Monera['Date']).dt.date
Monera = Monera.set_index('Date')



#Unique Database for all cyptos



Cryptos = pd.concat([Bitcoin, Ethereum, Monera], axis = 1)
Cryptos = Cryptos.dropna()
Cryptos = Cryptos.iloc[::-1]

Prices = Cryptos

Log_Prices = np.log(Prices)



#Graphical Representation



plt.figure(dpi = 1000, figsize=(16, 8))
plt.subplot(231)
plt.plot(Cryptos.iloc[:, 0], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Bitcoin/USD')
plt.xticks(rotation = 40)

plt.subplot(232)
plt.plot(Cryptos.iloc[:, 1], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Ethereum/USD Price')
plt.xticks(rotation = 40)
 
plt.subplot(233)
plt.plot(Cryptos.iloc[:, 2], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Monera/USD Price')
plt.xticks(rotation = 40)

plt.subplot(234)
plt.plot(Log_Prices.iloc[:, 0], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Bitcoin/USD Log-Price')
plt.xticks(rotation = 40)

plt.subplot(235)
plt.plot(Log_Prices.iloc[:, 1], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Ethereum/USD Log-Price')
plt.xticks(rotation = 40)

plt.subplot(236)
plt.plot(Log_Prices.iloc[:, 2], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Monera/USD Log-Price ')
plt.xticks(rotation = 40)
plt.tight_layout()
plt.show()



#Computation of Simple Returns



Returns_Simple = np.divide(Prices.iloc[1:1613, :], Prices.iloc[0:(1613 - 1), :]) - 1

plt.figure(dpi = 1000, figsize=(16, 5))
plt.subplot(131)
plt.plot(Returns_Simple.iloc[:, 0], color = 'red', linewidth = 1)
plt.ylabel('Returns')
plt.xlabel('Periods (Daily)')
plt.title('Bitcoin/USD Returns')
plt.xticks(rotation = 40)

plt.subplot(132)
plt.plot(Returns_Simple.iloc[:, 1], color = 'red', linewidth = 1)
plt.ylabel('Returns')
plt.xlabel('Periods (Daily)')
plt.title('Ethereum/USD Returns')
plt.xticks(rotation = 40)
 
plt.subplot(133)
plt.plot(Returns_Simple.iloc[:, 2], color = 'red', linewidth = 1)
plt.ylabel('Returns')
plt.xlabel('Periods (Daily)')
plt.title('Monera/USD Returns')
plt.xticks(rotation = 40)
plt.tight_layout()
plt.show()



#Correlation Matrix



plt.figure(dpi = 1000)
sns.heatmap(Prices.corr(), annot=True)
plt.title('Correlation Matrix Prices', fontdict={'fontsize':12}, pad=12)
plt.show()

plt.figure(dpi = 1000)
sns.heatmap(Log_Prices.corr(), annot=True)
plt.title('Correlation Matrix Log-Prices', fontdict={'fontsize':12}, pad=12)
plt.show()



#Characteristics for Simple Prices



Annualized_Mean = np.mean(Prices, 0)

Sigma = np.cov(np.transpose(Prices))

Volatility = np.power(np.diag(Sigma), 0.5)

Skewness = skew(Prices)

Kurtosis = kurtosis(Prices, fisher = False)

Maximum = Prices.max()

Minimum = Prices.min()



#Characteristics for Log Prices



Annualized_Mean2 = np.mean(Log_Prices, 0)

Sigma2 = np.cov(np.transpose(Log_Prices))

Volatility2 = np.power(np.diag(Sigma2), 0.5)

Skewness2 = skew(Log_Prices)

Kurtosis2 = kurtosis(Log_Prices, fisher = False)

Maximum2 = Log_Prices.max()

Minimum2 = Log_Prices.min()



#Density functions for all time series



plt.figure(dpi = 1000, figsize=(16, 8))
plt.subplot(231)
sns.distplot(Prices.iloc[:, 0], hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Prices.iloc[:, 0]) - 3*np.std(Prices.iloc[:, 0]), np.mean(Prices.iloc[:, 0]) + 3*np.std(Prices.iloc[:, 0]), 100)
plt.plot(x, norm.pdf(x, np.mean(Prices.iloc[:, 0]), np.std(Prices.iloc[:, 0])), color = 'black')
plt.legend(['PDF Prices', 'PDF Normal Distribution'])
plt.title('Density Functions Comparison for Bitcoin')
plt.xlabel('Bitcoin Simple Prices')

plt.subplot(232)
sns.distplot(Prices.iloc[:, 1], hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Prices.iloc[:, 1]) - 3*np.std(Prices.iloc[:, 1]), np.mean(Prices.iloc[:, 1]) + 3*np.std(Prices.iloc[:, 1]), 100)
plt.plot(x, norm.pdf(x, np.mean(Prices.iloc[:, 1]), np.std(Prices.iloc[:, 1])), color = 'black')
plt.legend(['PDF Prices', 'PDF Normal Distribution'])
plt.title('Density Functions Comparison for Ethereum')
plt.xlabel('Ethereum Simple Prices')

plt.subplot(233)
sns.distplot(Prices.iloc[:, 2], hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Prices.iloc[:, 2]) - 3*np.std(Prices.iloc[:, 2]), np.mean(Prices.iloc[:, 2]) + 3*np.std(Prices.iloc[:, 2]), 100)
plt.plot(x, norm.pdf(x, np.mean(Prices.iloc[:, 2]), np.std(Prices.iloc[:, 2])), color = 'black')
plt.legend(['PDF Prices', 'PDF Normal Distribution'])
plt.title('Density Functions Comparison for Monera')
plt.xlabel('Monera Simple Prices')

plt.subplot(234)
sns.distplot(Log_Prices.iloc[:, 0], hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Log_Prices.iloc[:, 0]) - 3*np.std(Log_Prices.iloc[:, 0]), np.mean(Log_Prices.iloc[:, 0]) + 3*np.std(Log_Prices.iloc[:, 0]), 100)
plt.plot(x, norm.pdf(x, np.mean(Log_Prices.iloc[:, 0]), np.std(Log_Prices.iloc[:, 0])), color = 'black')
plt.legend(['PDF Log Prices', 'PDF Normal Distribution'])
plt.title('Density Functions Comparison for Bitcoin')
plt.xlabel('Bitcoin Log Prices')

plt.subplot(235)
sns.distplot(Log_Prices.iloc[:, 1], hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Log_Prices.iloc[:, 1]) - 3*np.std(Log_Prices.iloc[:, 1]), np.mean(Log_Prices.iloc[:, 1]) + 3*np.std(Log_Prices.iloc[:, 1]), 100)
plt.plot(x, norm.pdf(x, np.mean(Log_Prices.iloc[:, 1]), np.std(Log_Prices.iloc[:, 1])), color = 'black')
plt.legend(['PDF Log Prices', 'PDF Normal Distribution'])
plt.title('Density Functions Comparison for Ethereum')
plt.xlabel('Ethereum Log Prices')

plt.subplot(236)
sns.distplot(Log_Prices.iloc[:, 2], hist=False, kde=True, bins=int(180/5), color = 'red', kde_kws={'linewidth': 1})
x = np.linspace(np.mean(Log_Prices.iloc[:, 2]) - 3*np.std(Log_Prices.iloc[:, 2]), np.mean(Log_Prices.iloc[:, 2]) + 3*np.std(Log_Prices.iloc[:, 2]), 100)
plt.plot(x, norm.pdf(x, np.mean(Log_Prices.iloc[:, 2]), np.std(Log_Prices.iloc[:, 2])), color = 'black')
plt.legend(['PDF Log Prices', 'PDF Normal Distribution'])
plt.title('Density Functions Comparison for Monera')
plt.xlabel('Monera Log Prices')
plt.tight_layout()



#=====================================================
#Part 1.1.1
#=====================================================



#Percentiles = [10, 5, 1, 0.1]
Parameters = []
Std_Errors = []


p = np.zeros((1613, 2))

for i in range(10000):
    Epsilon = np.random.normal(0, 1, 1613)
    p[:, 0] = Epsilon
    
    for j in range(1613):
        if j == 0:
            p[j, 1] = 0 
        else:
            p[j, 1] = p[j-1, 1] + p[j, 0]
    
    AR_Process = AutoReg(p[:, 1], 1, old_names=False)
    results = AR_Process.fit()
    Parameters.append(results._params[1])
    Std_Errors.append(results.bse[1])

Parameters = np.array(Parameters)
Std_Errors = np.array(Std_Errors)
One = np.ones(10000)
DF_Test = np.divide(np.subtract(Parameters, One), Std_Errors)


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1]

Critical_Values = ecdf(DF_Test)
s1 = Critical_Values[0][99]
s2 = Critical_Values[0][499]
s3 = Critical_Values[0][999]
print(s1)
print(s2)
print(s3)



#For the second model of the Random Walk 



Parameters2 = []
Std_Errors2 = []
p2 = np.zeros((1613, 2))

for i in range(10000):
    Epsilon2 = np.random.normal(0, 1, 1613)
    p2[:, 0] = Epsilon2
    
    for j in range(1613):
        if j == 0:
            p2[j, 1] = 0 
        else:
            p2[j, 1] = 0.2 * p2[j-1, 1] + p2[j, 0]
    
    AR_Process2 = AutoReg(p2[:, 1], 1, old_names=False)
    results2 = AR_Process2.fit()
    Parameters2.append(results2._params[1])
    Std_Errors2.append(results2.bse[1])

Parameters2 = np.array(Parameters2)
Std_Errors2 = np.array(Std_Errors2)
One2 = 0.2 * np.ones(10000)
DF_Test2 = np.divide(np.subtract(Parameters2, One2), Std_Errors2)

Critical_Values2 = ecdf(DF_Test2)
S1 = Critical_Values2[0][99]
S2 = Critical_Values2[0][499]
S3 = Critical_Values2[0][999]
print(S1)
print(S2)
print(S3)


#Graphical representation of the t-stats distribution
plt.figure(dpi = 1000)
plt.hist(DF_Test, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(DF_Test) - 3*np.std(DF_Test), np.mean(DF_Test) + 3*np.std(DF_Test), 100)
plt.plot(x, norm.pdf(x, np.mean(DF_Test), np.std(DF_Test)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram first model')
plt.xlabel('Dickey-Fuller t-stats')
plt.ylabel('Frequency')
plt.axvline(s1, linestyle = '--', linewidth = 1, color = 'darkred')
plt.text(0.99 * s1, 0.45,'1%', rotation = 90, color = 'darkred')
plt.axvline(s2, linestyle = '--', linewidth = 1, color = 'darkred')
plt.text(0.99 * s2, 0.45, '5%', rotation = 90, color = 'darkred')
plt.axvline(s3, linestyle = '--', linewidth = 1, color = 'darkred')
plt.text(0.99 * s3, 0.45, '10%', rotation = 90, color = 'darkred')
plt.show()

plt.figure(dpi = 1000)
plt.hist(DF_Test2, bins = "auto", rwidth=2, histtype= 'bar', density = True, color = 'white', stacked=True, edgecolor = 'red')
x = np.linspace(np.mean(DF_Test2) - 3*np.std(DF_Test2), np.mean(DF_Test2) + 3*np.std(DF_Test2), 100)
plt.plot(x, norm.pdf(x, np.mean(DF_Test2), np.std(DF_Test2)), color = 'darkred', linewidth = 1.5)
plt.legend(['Normal PDF'])
plt.title('Histogram second model')
plt.xlabel('Dickey-Fuller t-stats')
plt.ylabel('Frequency')
plt.axvline(S1, linestyle = '--', linewidth = 1, color = 'darkred')
plt.text(0.97 * S1, 0.35, '1%', rotation = 90, color = 'darkred')
plt.axvline(S2, linestyle = '--', linewidth = 1, color = 'darkred')
plt.text(0.97 * S2, 0.35, '5%', rotation = 90, color = 'darkred')
plt.axvline(S3, linestyle = '--', linewidth = 1, color = 'darkred')
plt.text(0.97 * S3, 0.35, '10%', rotation = 90, color = 'darkred')
plt.show()



#===========================================================
#Part 1.1.2
#===========================================================



AR_BTC = AutoReg(Log_Prices.iloc[:, 0], 1) #For Bitcoin
Results_BTC = AR_BTC.fit()
Param_BTC = Results_BTC.params
Std_Error_BTC = Results_BTC.bse
t_Stat_BTC = np.divide(np.subtract(Param_BTC[1], 1), Std_Error_BTC[1])

AR_ETH = AutoReg(Log_Prices.iloc[:, 1], 1) #For ETH
Results_ETH = AR_ETH.fit()
Param_ETH = Results_ETH.params
Std_Error_ETH = Results_ETH.bse
t_Stat_ETH = np.divide(np.subtract(Param_ETH[1], 1), Std_Error_ETH[1])

AR_XMR = AutoReg(Log_Prices.iloc[:, 2], 1) #For XMR
Results_XMR = AR_XMR.fit()
Param_XMR = Results_XMR.params
Std_Error_XMR = Results_XMR.bse
t_Stat_XMR = np.divide(np.subtract(Param_XMR[1], 1), Std_Error_XMR[1])


X_BTC = sm.add_constant(Log_Prices.iloc[:(len(Log_Prices) - 1), 0])
X_ETH = sm.add_constant(Log_Prices.iloc[:(len(Log_Prices) - 1), 1])
X_XMR = sm.add_constant(Log_Prices.iloc[:(len(Log_Prices) - 1), 2])

plt.figure(dpi = 1000, figsize=(16, 5))
plt.subplot(131)
plt.scatter(Log_Prices.iloc[:(len(Log_Prices) - 1), 0], Log_Prices.iloc[1:, 0], s = 1, color = 'red')
plt.plot(X_BTC, Param_BTC[0] + Param_BTC[1] * X_BTC, color = 'blue', linewidth = 1)
plt.axis([6.5, 11.5, 6.5, 11.5])
plt.xlabel('Log-Prices at t-1')
plt.ylabel('Log-Prices at t')
plt.title('Regression for Bitcoin')
plt.legend(["Regression"])

plt.subplot(132)
plt.scatter(Log_Prices.iloc[:(len(Log_Prices) - 1), 1], Log_Prices.iloc[1:, 1], s = 1, color = 'red')
plt.plot(X_ETH, Param_ETH[0] + Param_ETH[1] * X_ETH, color = 'blue', linewidth = 1)
plt.axis([3.5, 8.5, 3.5, 8.5])
plt.xlabel('Log-Prices at t-1')
plt.ylabel('Log-Prices at t')
plt.title('Regression for Ethereum')
plt.legend(["Regression"])

plt.subplot(133)
plt.scatter(Log_Prices.iloc[:(len(Log_Prices) - 1), 2], Log_Prices.iloc[1:, 2], s = 1, color = 'red')
plt.plot(X_XMR, Param_XMR[0] + Param_XMR[1] * X_XMR, color = 'blue', linewidth = 1)
plt.axis([2.5, 6.5, 2.5, 6.5])
plt.xlabel('Log-Prices at t-1')
plt.ylabel('Log-Prices at t')
plt.title('Regression for Monera')
plt.legend(["Regression"])
plt.tight_layout()



#===========================================
#Part 1.2
#===========================================



#Creation of a function that show if a function is stationary or not
def Statio(x):
    diff_x = np.subtract(x[1:], x[0:(len(x) - 1)])
    X0 = sm.add_constant(x[:(len(x) - 1)])
    Regression_Resid = sm.OLS(diff_x, X0)
    Results_Resid = Regression_Resid.fit()
    Param_Resid = Results_Resid.params
    Std_Error_Resid = Results_Resid.bse
    t_Stat_Resid = np.divide(np.subtract(Param_Resid[1], 0), Std_Error_Resid[1])
    plt.plot(diff_x)
    plt.show()
    print(t_Stat_Resid)
    print(Param_Resid)
    

#Regression BTC-ETH
X1 = sm.add_constant(Log_Prices.iloc[:, 1])
BTC_ETH_Reg = sm.OLS(Log_Prices.iloc[:, 0], X1)
Results11 = BTC_ETH_Reg.fit()
Para_BTC_ETH = Results11.params
Z_1_Resid = Results11.resid
Z_1_Resid = np.array(Z_1_Resid)


#Regression XMR-BTC
X2 = sm.add_constant(Log_Prices.iloc[:, 0])
XMR_BTC_Reg = sm.OLS(Log_Prices.iloc[:, 2], X2)
Results12 = XMR_BTC_Reg.fit()
Para_XMR_BTC = Results12.params
Z_2_Resid = Results12.resid
Z_2_Resid = np.array(Z_2_Resid)


#Regression XMR-ETH
X3 = sm.add_constant(Log_Prices.iloc[:, 1])
XMR_ETH_Reg = sm.OLS(Log_Prices.iloc[:, 2], X3)
Results13 = XMR_ETH_Reg.fit()
Para_XMR_ETH = Results13.params
Z_3_Resid = Results13.resid
Z_3_Resid = np.array(Z_3_Resid)
plt.plot(Z_3_Resid)


#Graphical Representation
Log_Prices_Array = np.array(Log_Prices)
X1 = np.array(X1)
X2 = np.array(X2)
X3 = np.array(X3)


plt.figure(dpi = 1000, figsize=(16, 5))
plt.subplot(131)
plt.scatter(X1[:, 1], Log_Prices_Array[:, 0], s = 1, color = 'red')
plt.plot(X1, Para_BTC_ETH[0] + Para_BTC_ETH[1] * X1, color = 'blue', linewidth = 1)
plt.axis([1.5, 8.5, 6, 11.5])
plt.xlabel('Ethereum Log-Prices')
plt.ylabel('Bitcoin Log-Prices')
plt.title('Regression of Bitcoin on Ethereum')
plt.legend(["Regression"])

plt.subplot(132)
plt.scatter(X2[:, 1], Log_Prices_Array[:, 2], s = 1, color = 'red')
plt.plot(X2, Para_XMR_BTC[0] + Para_XMR_BTC[1] * X2, color = 'blue', linewidth = 1)
plt.axis([6, 11.5, 1.5, 6.5])
plt.xlabel('Bitcoin Log-Prices')
plt.ylabel('Monera Log-Prices')
plt.title('Regression of Monera on Bitcoin')
plt.legend(["Regression"])

plt.subplot(133)
plt.scatter(X3[:, 1], Log_Prices_Array[:, 2], s = 1, color = 'red')
plt.plot(X3, Para_XMR_ETH[0] + Para_XMR_ETH[1] * X3, color = 'blue', linewidth = 1)
plt.axis([1.5, 8.5, 1.5, 6.5])
plt.xlabel('Ethereum Log-Prices')
plt.ylabel('Monera Log-Prices')
plt.title('Regression of Monera on Ethereum')
plt.legend(["Regression"])
plt.tight_layout()


#Check for stationarity
Statio(Z_1_Resid)
Statio(Z_2_Resid)
Statio(Z_3_Resid)

diff_Z_1 = np.subtract(Z_1_Resid[1:], Z_1_Resid[0:(len(Z_1_Resid) - 1)])
plt.plot(Z_3_Resid)


#Check cointegration with visualization
plt.figure(dpi = 1000)
plt.plot(Log_Prices.iloc[:, 1], color = 'red', linewidth = 1)
plt.plot(Log_Prices.iloc[:, 2], color = 'darkred', linewidth = 1)
plt.plot(Log_Prices.iloc[:, 0], color = 'red', linewidth = 1)
plt.xlabel('Periods (Daily)')
plt.ylabel('Log-Prices')
plt.title('Log-Prices of Monera and Ethereum')
plt.legend(["Ethereum", "Monera"])
plt.xticks(rotation = 40)
plt.show()



#=================================================
#Pair trading
#=================================================



#Computation of the returns
Log_Returns_XMR =  np.subtract(Log_Prices_Array[1:1613, 2], Log_Prices_Array[0:(1613 - 1), 2])
Log_Returns_ETH =  np.subtract(Log_Prices_Array[1:1613, 1], Log_Prices_Array[0:(1613 - 1), 1])


#Computation of the Spread
Spread = np.subtract(Log_Returns_XMR, Log_Returns_ETH)
Spread_Bar = np.mean(Spread)
Volat_Spread = np.mean(np.power(Spread, 2)) - np.power(Spread_Bar, 2)

#Computation of the Z-score
z_score = np.divide(np.subtract(Spread, Spread_Bar), Volat_Spread)

z_upper = np.quantile(z_score, 0.95)
z_lower = np.quantile(z_score, 0.05)
z_out = np.quantile(z_score, 0.40)

#Graphical Representation of the signals
Open1 = []
Open2 = []
for i in range(len(z_score)):
    if z_score[i] > z_upper:
        Open1.append([z_score[i], Log_Prices.index[i+1]])
    elif z_score[i] < z_lower:
        Open2.append([z_score[i], Log_Prices.index[i+1]])
        
Open1 = np.array(Open1)
Open2 = np.array(Open2)

plt.figure(dpi = 1000)
plt.plot(Log_Prices.index[1:], z_score, color = 'darkred', linewidth = 0.5)
plt.axhline(z_lower, color = 'red', linestyle = '--', linewidth = 0.8)
plt.axhline(z_upper, color = 'blue', linestyle = '--', linewidth = 0.8)
plt.plot(Open1[:, 1], Open1[:, 0], color = 'green', linestyle = 'None', marker = 'o', markersize = 2.5)
plt.plot(Open2[:, 1], Open2[:, 0], color = 'red', linestyle = 'None', marker = 'o', markersize = 2.5)
plt.title('Evolution of the z-scores')
plt.xlabel('Periods (Daily)')
plt.ylabel('Value of the z-scores')
plt.legend(['z-scores', 'z-lower', 'z-upper'])
plt.xticks(rotation = 40)

z_score_higher = z_score[z_score > z_upper]
len(z_score_higher)
z_score_lower = z_score[z_score < z_lower]
len(z_score_lower)

#Start of the Wealth
Wealth = np.zeros((1612))

#Array of Portfolio Returns
Portfolio_Returns = np.zeros((1612))

#Matrix of weight
Weights = np.zeros((1612, 2))

#Computation of the algorithm
LongA = []
LongB = []

for t in range(len(z_score)): #Different situations:
    
    #If nothing happens
    if len(LongA) == 0 and z_lower <  z_score[t] < z_upper and len(LongB) == 0:
        
        if t == 0:
            Wealth[0] = 100 #Initialize the first wealth at 100
        else:
            Wealth[t] = Wealth[t - 1] #If nothing happens at t, the wealth is the same as yesterday
    
    #Signal to open: Long B and Short A
    elif z_score[t] > z_upper and len(LongB) == 0 and len(LongA) == 0:
        
        Weights[t, 0] = 0.5
        Weights[t, 1] = 0.5
        
        Wealth[t] = Wealth[t - 1]
        LongB.append(1)
        
    #Nothing happens after opening our position but we adjust our wealth with the price evolution
    elif z_score[t] > z_out and len(LongB) > 0 and len(LongA) == 0:
        
        Weights[t, 0] = (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t])) / (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t]))
        Weights[t, 1] = (Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t])) / (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t]))
        
        Wealth[t] = Wealth[t - 1] * (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t]))
        Portfolio_Returns[t] = np.log((Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t])))

    #If we close the positions of Long B and Short A
    elif z_score[t] <= z_out and len(LongB) > 0 and len(LongA) == 0:
        
        Weights[t, 0] = (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t])) / (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t]))
        Weights[t, 1] = (Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t])) / (Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t]))
        
        Wealth[t] = Wealth[t - 1] * ((Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t])))
        Portfolio_Returns[t] = np.log((Weights[t - 1, 0] * math.exp((-1) * Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp(Log_Returns_ETH[t])))
        LongB = []
        
    #Signal to open: Long A and Short B
    elif z_score[t] < z_lower and len(LongA) == 0 and len(LongB) == 0:
        
        Weights[t, 0] = 0.5
        Weights[t, 1] = 0.5
        
        Wealth[t] = Wealth[t - 1]
        LongA.append(1)

    #Nothing happens after opening but we adjust our wealth with the price evolution
    elif z_score[t] < (- 1) * z_out and len(LongA) > 0 and len(LongB) == 0:
        
        Weights[t, 0] = (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t])) / (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) +  Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        Weights[t, 1] = (Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t])) / (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        
        Wealth[t] = Wealth[t - 1] * (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        Portfolio_Returns[t] = np.log(Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) +  Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        
    #If we close the positions of Long A and Short B
    elif z_score[t] >= (- 1) * z_out and len(LongA) > 0 and len(LongB) == 0:
        
        Weights[t, 0] = (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t])) / (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        Weights[t, 1] = (Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t])) / (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        
        Wealth[t] = Wealth[t - 1] * (Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) + Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        Portfolio_Returns[t] = np.log(Weights[t - 1, 0] * math.exp(Log_Returns_XMR[t]) +  Weights[t - 1, 1] * math.exp((-1) * Log_Returns_ETH[t]))
        LongA = []

plt.figure(dpi = 1000)
plt.plot(Log_Prices.index[1:], Wealth, linewidth = 1, color = 'red')
plt.xticks(rotation = 40)
plt.title('Evolution of the Wealth Performance')
plt.xlabel('Periods (Daily)')
plt.ylabel('Value of our Wealth')
plt.show()

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
sns.scatterplot(Log_Prices.index[1:], Weights[:, 0], s = 5, color = 'red')
plt.xticks(rotation = 40)
plt.title('Evolution of the Weights for Monera')
plt.xlabel('Periods (Daily)')
plt.ylabel('Weights')

plt.subplot(122)
sns.scatterplot(Log_Prices.index[1:], Weights[:, 1], s = 5, color = 'red')
plt.xticks(rotation = 40)
plt.title('Evolution of the Weights for Ethereum')
plt.xlabel('Periods (Daily)')
plt.ylabel('Weights')
plt.show()

plt.figure(dpi = 1000)
plt.plot(Log_Prices.index[1:], Portfolio_Returns, linewidth = 1, color = 'red')
plt.xticks(rotation = 40)
plt.title('Portfolio Log-Returns')
plt.xlabel('Periods (Daily)')
plt.ylabel('Log-Returns')
plt.show()

Annualized_Mean12 = np.mean(Portfolio_Returns, 0) * 252

Volatility12 = np.std(Portfolio_Returns) * np.power(252, 0.5)

Skewness12 = skew(Portfolio_Returns)

Kurtosis12 = kurtosis(Portfolio_Returns, fisher = False)

Maximum12 = Portfolio_Returns.max()

Minimum12 = Portfolio_Returns.min()


Annualized_Mean22 = np.mean(Wealth, 0)

Volatility22 = np.std(Wealth)

Skewness22 = skew(Wealth)

Kurtosis22 = kurtosis(Wealth, fisher = False)

Maximum22 = Wealth.max()

Minimum22 = Wealth.min()




#=============================================================
#Other Stuff (We don't used this in our previous computations)
#=============================================================



def Stat(x):
    AR_Resid = AutoReg(x, 1)
    Results_Resid = AR_Resid.fit()
    Param_Resid = Results_Resid.params
    Std_Error_Resid = Results_Resid.bse
    t_Stat_Resid = np.divide(np.subtract(Param_Resid[1], 1), Std_Error_Resid[1])
    plt.plot(x)
    plt.show()
    print(t_Stat_Resid)
    print(Param_Resid)


Stat(Z_1_Resid)
Stat(Z_2_Resid)
Stat(Z_3_Resid)


from statsmodels.tsa.stattools import adfuller
def stationarity(a, cutoff = 0.10):
  a = np.ravel(a)
  if adfuller(a)[1] < cutoff:
    print('‘The series is stationary’')
    print('‘p-value = ‘', adfuller(a)[1])
  else:
    print('‘The series is NOT stationary’')
    print('‘p-value = ‘', adfuller(a)[1])

stationarity(Z_3_Resid)



