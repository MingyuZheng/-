# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:54:13 2019

@author: DELL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as scio
import datetime
from sklearn import linear_model
from sklearn.metrics import r2_score

# In[1] 数据导入及简单变量生成
orig=scio.loadmat('IF2018.mat')
data_orig=orig['IFdata']
df=pd.DataFrame(data_orig)
df.columns =['day','time','close','open','high','low','bid','ask','bid_vol','ask_vol','turnover','pos']
df['spread']=df['ask']-df['bid']
df['OIR']=(df['bid_vol']-df['ask_vol'])/df['bid_vol']
df['mid_price']=(df['ask']+df['bid'])/2
data=df.values

# In[2] 构建变量函数

def get_data_daily(date,data):
    daily=data[data[:,0]==date,:] #构建每天的数据集
    #构建变量时若上下午时间不用分开，则不需分开产生上下午数据
    return daily



def delta_V(data1,data2,n): 
    #需要单独的ask ask_vol 和 bid  bid_vol列作为输入值,n为单独每一天的数据长度
    #不需要分上下午
    delta_v=[]
    delta_v.append(data2[0])
    for i in range(1,n):
        if data1[i]<=data1[i-1]:
            delta_v.append(0)
        elif data1[i]==data1[i-1]:
            delta_v.append(data2[i]-data2[i-1])
        else:
            delta_v.append(data2[i])
    return delta_v
# VOI=delta_vb-delta_va



def var_lag(var,lag,spread,n):
    #构建回归需要滞后项的变量
    #lag为滞后阶数
    #n为长度
    X1=np.array([var[i]/spread[i] for i in range(n)]).reshape(n,1)
    df_var=pd.DataFrame(var)
    for i in range(1,lag+1):
        x1=df_var.shift(periods=i).values
        if i==1:
            x1[0]=var[0]
        else:
            x1[0:i]=var[0]
        x11=np.array([x1[i]/spread[i] for i in range(n)]).reshape(n,1)
        X1=np.hstack((X1,x11)) #X1,x11为array
    return X1

    
    
def linear_model_main(voi,spread,oir,mid_price,lag,k,n): 
    #前5个变量全为array
    #lag为时间序列滞后阶数，k为价格变化预测窗口
    #回归变量X为array类型，每一个x变量为一列
    Y=[]
    for i in range(len(mid_price)):
        if i > len(mid_price)-21 and i< len(mid_price)-1 :
            Y.append(mid_price[i+1:len(mid_price)].mean())
        elif i==len(mid_price)-1:
            Y.append(mid_price[len(mid_price)-1])
        else:
            Y.append(mid_price[i+1:i+20].mean()) 
            
    delata_M=np.array([Y[i]-mid_price[i] for i in range(n)])
    X1=var_lag(voi,lag,spread,n)
    X2=var_lag(oir,lag,spread,n)
    X=np.hstack((X1,X2))
    regr = linear_model.LinearRegression()
    regr.fit(X, delata_M)
    delata_M_pred=regr.predict(X)
    coef=regr.coef_
    inter=regr.intercept_ 
    r2=r2_score(delata_M,delata_M_pred)
    return delata_M_pred,r2,coef,inter


    

# In[3] 计算全年平均回归系数
        
def predict_year(data):  
    trans_day=np.unique(data[:,0])           #记录一年内所有交易日
    trans_day.size                 #计算交易日
    inter_year=[]
    for i in range(0,243):            #生成所有交易日的上下午数据
        date=trans_day[i]
        daily=get_data_daily(date,data)
        ask=daily[:,7]
        ask_vol=daily[:,9]
        bid=daily[:,6]
        bid_vol=daily[:,8]
        OIR=daily[:,13]
        mid_price=daily[:,14]
        spread=daily[:,12]
        n=daily.shape[0]
        lag=5     #定义数据滞后阶数
        k=20     #预测窗口长度    
        delta_vb=delta_V(bid,bid_vol,n)
        delta_va=delta_V(ask,ask_vol,n)
        VOI=[delta_vb[i]-delta_va[i] for i in range(n)]
        outcome=linear_model_main(VOI,spread,OIR,mid_price,lag,k,n)
        predict_MP=outcome[0]
        R2=outcome[1]
        coef=outcome[2]
        inter=outcome[3]
        
        predict_MP_year=pd.DataFrame()
        predict_MP_year[date]=predict_MP.tolist()
        
        coef_year=pd.DataFrame()
        g=np.transpose(coef).reshape(12,).tolist()
        coef_year[date]=g
        
        inter_year.append(inter)
    avg=[]
    for j in range(13):
         if j==0:
            inter_avg=np.mean(inter_year)
            avg.append(inter_avg)
         else:
            b=coef_year.iloc[j-1,:].mean()
            avg.append(b)
        
    coef_avg_year=pd.DataFrame()
    coef_avg_year['coef_name']=['intercept','VOI','VOI1','VOI2','VOI3','VOI4','VOI5','OIR','OIR1','OIR2','OIR3','OIR4','OIR5']
    coef_avg_year['coef_values']=avg
    return coef_avg_year
        
        
# In[4]策略结果重现
print(predict_year(data)) 

def linear_daily(daily):
    ask=daily[:,7]
    ask_vol=daily[:,9]
    bid=daily[:,6]
    bid_vol=daily[:,8]
    OIR=daily[:,13]
    mid_price=daily[:,14]
    spread=daily[:,12]
    n=daily.shape[0]
    lag=5     #定义数据滞后阶数
    k=20     #预测窗口长度    
    delta_vb=delta_V(bid,bid_vol,n)
    delta_va=delta_V(ask,ask_vol,n)
    VOI=[delta_vb[i]-delta_va[i] for i in range(n)]
    outcome=linear_model_main(VOI,spread,OIR,mid_price,lag,k,n)
    return outcome
        
def profit_daily(data):  
    profits=0 #每次交易的收益
    total_pnl=[] #每天profit
    trade_volumn=[] #每天成交量
    num=0 #交易量
    buy_price=0 #成交时的买入价格
    sell_price=0 #成交时卖出价格
    trans_day=np.unique(data[:,0])           #记录一年内所有交易日
    trans_day.size                 #计算交易日
    for i in range(0,10):            #生成所有交易日的上下午数据
        date=trans_day[i]
        daily=get_data_daily(date,data)
        ask=daily[:,7]
        bid=daily[:,6]
        outcome=linear_daily(daily)
        X=outcome[0]
        n=daily.shape[0]
        if i>0:
            date=trans_day[i-1]
            daily1=get_data_daily(date,data) #用t-1天数据确定回归系数
            outcome1=linear_daily(daily1)
            coef=outcome1[2]
            inter=outcome1[3]
            c=np.matrix(coef)
            for j in range(n):
                v=np.matrix(X[j,:]).T
                signal=c*v+inter
                if signal>0.2:
                    buy_price=ask[j]
                    num+=1
                    profits+=sell_price-buy_price
                elif signal<-0.2:
                    sell_price=bid[j]
                    num+=-1
                    profits+=sell_price-buy_price
                else:
                    pass
            trade_volumn.append(num) 
            total_pnl.append(profits)
               
        else:
            pass
    return total_pnl,trade_volumn

temp=profit_daily(data)  
pl=temp[0]
print(pl)

#画100天策略日收益率图

plt.show()
plt.plot(pl)

