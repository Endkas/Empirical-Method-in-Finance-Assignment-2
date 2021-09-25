#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:35:21 2021

@author: alessio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import numpy.matlib
from scipy.stats import multivariate_normal
from scipy.stats.distributions import chi2
from statsmodels.tsa.stattools import kpss
import seaborn as sns



#============================================================
#Download of the data
#============================================================



filename = "Part2_dataset.xlsx"

xls = pd.ExcelFile(filename)

Data = pd.read_excel(xls, 'Sheet1')

Data['Date'] = pd.to_datetime(Data['Unnamed: 0']).dt.date
Data = Data.set_index('Date')



#=============================================================
#Database of each price time series
#=============================================================



Price = Data.iloc[:, 1:5]

Log_Price = np.log(Price)



#=============================================================
#Graphical Representation
#=============================================================



plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Price.iloc[:, 0], color = 'red', linewidth = 0.9)
plt.ylabel('Dollar Index')
plt.xlabel('Periods (Daily)')
plt.title('Price of US Dollar in trade weighted terms')

plt.subplot(122)
plt.plot(Log_Price.iloc[:, 0], color = 'red', linewidth = 0.9)
plt.ylabel('Dollar Index')
plt.xlabel('Periods (Daily)')
plt.title('Log-Price of US Dollar in trade weighted terms')
plt.show()

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Price.iloc[:, 1], color = 'red', linewidth = 0.9)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Gold Price')

plt.subplot(122)
plt.plot(Log_Price.iloc[:, 1], color = 'red', linewidth = 0.9)
plt.ylabel('Log-Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('Gold Log-Price')
plt.show()

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Price.iloc[:, 2], color = 'red', linewidth = 0.9)
plt.ylabel('Index')
plt.xlabel('Periods (Daily)')
plt.title('VIX Index')

plt.subplot(122)
plt.plot(Log_Price.iloc[:, 2], color = 'red', linewidth = 0.9)
plt.ylabel('Index')
plt.xlabel('Periods (Daily)')
plt.title('VIX Index (in log)')
plt.show()

plt.figure(dpi = 1000, figsize=(16, 4))
plt.subplot(121)
plt.plot(Price.iloc[:, 3], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('BTC/USD Price')

plt.subplot(122)
plt.plot(Log_Price.iloc[:, 3], color = 'red', linewidth = 1)
plt.ylabel('Price in Dollars')
plt.xlabel('Periods (Daily)')
plt.title('BTC/USD Log-Price')
plt.show()


plt.figure(dpi = 1000)
sns.heatmap(Price.corr(), annot=True)
plt.title('Correlation Matrix Prices', fontdict={'fontsize':12}, pad=12)
plt.show()

plt.figure(dpi = 1000)
sns.heatmap(Log_Price.corr(), annot=True)
plt.title('Correlation Matrix Log-Prices', fontdict={'fontsize':12}, pad=12)
plt.show()



#====================================================================
#Question 2: Testing Stationarity
#====================================================================



def adf_test(timeseries): #Augmented Dickey-Fuller Test
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
       dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(Log_Price.iloc[:,0])
adf_test(Log_Price.iloc[:,1])
adf_test(Log_Price.iloc[:,2])
adf_test(Log_Price.iloc[:,3])


def kpss_test(timeseries): #KPSS Test to complete the precedent test
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)
    
kpss_test(Log_Price.iloc[:,0])
kpss_test(Log_Price.iloc[:,1])
kpss_test(Log_Price.iloc[:,2])
kpss_test(Log_Price.iloc[:,3])



#====================================================
#Creating Stationary processes
#====================================================



Log_Prices_Stat = np.subtract(Log_Price.iloc[1:, :], Log_Price.iloc[0:(len(Log_Price) - 1), :])

plt.figure(dpi = 1000, figsize=(16, 8))
plt.subplot(221)
plt.plot(Log_Prices_Stat.iloc[:, 0], color = 'red', linewidth = 0.6)
plt.ylabel('Dollar Index Returns')
plt.xlabel('Periods (Daily)')
plt.title('US Dollar Log-Returns')

plt.subplot(222)
plt.plot(Log_Prices_Stat.iloc[:, 1], color = 'red', linewidth = 0.6)
plt.ylabel('Log-Returns')
plt.xlabel('Periods (Daily)')
plt.title('Gold Log-Returns')
 
plt.subplot(223)
plt.plot(Log_Prices_Stat.iloc[:, 2], color = 'red', linewidth = 0.6)
plt.ylabel('Index')
plt.xlabel('Periods (Daily)')
plt.title('VIX Index')

plt.subplot(224)
plt.plot(Log_Prices_Stat.iloc[:, 3], color = 'red', linewidth = 0.6)
plt.ylabel('Log-Returns')
plt.xlabel('Periods (Daily)')
plt.title('BTC/USD Log-Returns')
plt.tight_layout()
plt.show()

adf_test(Log_Prices_Stat.iloc[:,0])
adf_test(Log_Prices_Stat.iloc[:,1])
adf_test(Log_Prices_Stat.iloc[:,2])
adf_test(Log_Prices_Stat.iloc[:,3])

kpss_test(Log_Prices_Stat.iloc[:,0])
kpss_test(Log_Prices_Stat.iloc[:,1])
kpss_test(Log_Prices_Stat.iloc[:,2])
kpss_test(Log_Prices_Stat.iloc[:,3])



#===============================================
#Question 3: VAR Model estimation
#===============================================



model = VAR(Log_Prices_Stat)
x = model.select_order(12)
print(x.summary()) #We choose a lag of one

#Other Method
for i in [1,2,3,4,5,6,7,8,9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

model_fitted = model.fit(1)
model_fitted.summary()

Covar = model_fitted.bse

Phi1 = model_fitted.coefs;
Phi0 = model_fitted.coefs_exog;
theta = [*Phi1.flatten(),*Phi0.flatten()]


#Impulse Reaction function

#### 3 Impulse
Cov = model_fitted.sigma_u
cov = np.array(Cov)
s = [cov[0,0]**0.5,cov[1,1]**0.5,cov[2,2]**0.5] 
#p0 = np.array([cov[0,0]**0.5,cov[1,1]**0.5,cov[2,2]**0.5,0]) 
shocks = []
for i in range(len(s)):
    p0 = np.zeros((4,1))
    p0[i] = s[i]
    print(p0)
    p = np.add(Phi0,p0)
    phi1 = np.reshape(Phi1,(4,4))
    pp = [p]
    for i in range(1,6) :
        q = Phi0 + np.matmul(phi1,pp[i-1])
        pp.append(q)
        Bp = []
        for i in pp :
            Bp.append(i[3])
    shocks.append(Bp)

p0 = np.array([cov[0,0]**0.5,cov[1,1]**0.5,cov[2,2]**0.5,0]) 
p0 = np.reshape(p0,(4,1))
p = np.add(Phi0,p0)
phi1 = np.reshape(Phi1,(4,4))
pp = [p]
for i in range(1,6) :
    q = Phi0 + np.matmul(phi1,pp[i-1])
    pp.append(q)
    Bp = []
for i in pp :
    Bp.append(i[3])
shocks.append(Bp)


#Graphical Representation of the shocks    
plt.figure(dpi = 1000, figsize=(12, 6))
plt.subplot(221)
plt.plot(shocks[0], color = 'red', linewidth = 1.1)
plt.ylabel('Bitcoin return')
plt.title('Impact of a 1 std shock of the Dollar on the BitCoin.')

plt.subplot(222)
plt.plot(shocks[1], color = 'red', linewidth = 1.1)
plt.ylabel('Bitcoin return')
plt.title('Impact of a 1 std shock of the Gold on the BitCoin.')

plt.subplot(223)
plt.plot(shocks[2], color = 'red', linewidth = 1.1)
plt.ylabel('Bitcoin return')
plt.xlabel('Periods (Daily)')
plt.title('Impact of a 1 std shock of the VIX on the BitCoin.')

plt.subplot(224)
plt.plot(shocks[3], color = 'red', linewidth = 1.1)
plt.ylabel('Bitcoin return')
plt.xlabel('Periods (Daily)')
plt.title('Combined shocks on the BitCoin.')
print(Phi0[3])
plt.tight_layout()


irf = model_fitted.irf(5)
irf.plot(orth=False)

fig = irf.plot(response=3)
fig.tight_layout()
fig.set_figheight(9)
fig.set_figwidth(8)



#==================================================
#Question 4: 
#==================================================
    


def ML_VAR(theta,X):
    l1 =[theta[0], theta[1], theta[2],theta[3]]
    l2 =[theta[4], theta[5],theta[6], theta[7]]
    l3 =[theta[8], theta[9],theta[10], theta[11]]
    l4 = [theta[12], theta[13], theta[14],theta[15]]
    l5 = [theta[16], theta[17], theta[18],theta[19]]
    Phi1=[l1,l2,l3,l4]
    Phi0=[l5]
    Y=X.iloc[1:np.size(X,0),:]
    Z=X.iloc[0:(np.size(X,0)-1),:]
    Phi0_temp= np.matlib.repmat(Phi0,np.size(Z,0),1)
    temp=np.matmul(Z,Phi1)-Phi0_temp
    res=np.subtract(Y,temp)
    loglik=multivariate_normal.logpdf(res, mean=[0,0,0,0], cov=np.identity(4))
    loglik=-np.sum(loglik)
    return loglik


Constraint = ({'type':'eq', 'fun': lambda x: x[3]}, 
              {'type':'eq', 'fun': lambda x: x[7]},
              {'type':'eq', 'fun': lambda x: x[11]})

estimation_output2 = minimize(ML_VAR, theta, method='SLSQP', constraints = Constraint, args=(Log_Prices_Stat)) #Constrained model
estimated_para2 = estimation_output2.x
Cons_Likelihood = (- 1) * estimation_output2.fun

Phi1 = model_fitted.coefs;
Phi0 = model_fitted.coefs_exog;
theta = [*Phi1.flatten(),*Phi0.flatten()]

estimation_output = minimize(ML_VAR, theta, method='SLSQP', args=(Log_Prices_Stat)) #Unconstrained model
estimated_para = estimation_output.x
Uncons_Likelihood = (- 1) * estimation_output.fun



#============================================================
#Likelihood Test
#============================================================



LR_Test = 2 * (Uncons_Likelihood - Cons_Likelihood)
p_value = chi2.sf(LR_Test, 3)




