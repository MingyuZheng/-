# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:48:50 2019

@author: fksx2
"""

import cx_Oracle
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import math
from sklearn import linear_model

#连接数据库
def connect_database():
    conn = cx_Oracle.connect('jcquery','83936666','10.88.102.160:1521/gfdwdb1')
    return conn

#数据导入 指数 医药基金 基金名单
def index(conn):    
    sql1 = '''
            --申万医药生物指数
    select i.S_INFO_WINDCODE,i.TRADE_DT,i.S_DQ_CLOSE,(i.S_DQ_CLOSE/i.l-1) r from
    (
    select S_INFO_WINDCODE,TRADE_DT,S_DQ_CLOSE,lag(S_DQ_CLOSE) over (order by TRADE_DT) l
    from gfwind.ASWSIndexEOD 
    where S_INFO_WINDCODE='801150.SI'
    and TRADE_DT>20141230
    ) i
          '''
    ind=pd.read_sql(sql1,conn)
    return ind

def input_name(file):
    df=pd.read_excel(file)
    df1=df['WIND_CODE'].values
    return df1

def ind_fund(fund_name,conn):
    sql2='''
    select f.F_INFO_WINDCODE,f.PRICE_DATE,f.F_NAV_UNIT,(f.F_NAV_UNIT/f.l-1) rf,null zd from
    (
    SELECT F_INFO_WINDCODE，	PRICE_DATE，F_NAV_UNIT，lag(F_NAV_UNIT) over (order by PRICE_DATE) l
    FROM GFWIND.ChinaMutualFundNAV
    where F_INFO_WINDCODE = '%s'
    ) f
    '''%(fund_name)
    f=pd.read_sql(sql2,conn)
    return f

#划分涨跌区间
def split(df):
    

#记录各区间每个基金的表现
#z 涨跌区间端点对应index记录 ; fund 每个基金的数据 
    
def profit(fund,z,date):
    j=0
    net=[]
    while j<(len(z)-1):
        begin=z[j]
        end=z[j+1]
        b=fund[fund['PRICE_DATE']==date[begin]].index.tolist()
        e=fund[fund['PRICE_DATE']==date[end]].index.tolist()
        if b!=[] and e!=[]:
            net.append(fund['F_NAV_UNIT'][e[0]]/fund['F_NAV_UNIT'][b[0]]-1)
        else:
            net.append(np.nan)
        j=j+2
    return net

#医药各类型下排序和打分
def order_score(total,name,z,date,conn):
    order=[]
    for i in range(total):
        fund=ind_fund(name[i],conn)
        net=profit(fund,z,date)   
        order.append(net)   
    order=pd.DataFrame(order)
    order.index=name
    n=int(len(z)/2)
    
    score1=[]
    for j in range(n):
        a=order.iloc[:,j].sort_values(ascending=False)
        count=0
        for i in range(total):
            if a.iloc[i]>=0 or a.iloc[i]<0:
                count=count+1
        s=math.ceil(count/5)      
        for i in range(total):
            if i<s:
                a.iloc[i]=5
            elif i>=s and i<2*s:
                a.iloc[i]=4
            elif i>=2*s and i<3*s:
                a.iloc[i]=3
            elif i>=3*s and i<4*s:
                a.iloc[i]=2
            elif i>=4*s and i<count:
                a.iloc[i]=1
            else:
                a.iloc[i]=np.nan
        score1.append(a) 
    net1=pd.DataFrame(score1)
    net1.loc['mean']=0
    for i in range (total):
        net1.loc['mean',net1.columns[i]]=net1.iloc[:n,i].mean()    
    score2=net1.loc['mean']
    return score2

#择时能力和选股能力并打分
def reg(fund,df): 
    fund=fund.drop(['ZD'],axis=1)
    fund=fund.dropna()
    df=df.dropna()
    X=[]
    a=pd.DataFrame()
    for i in range(fund.shape[0]):
        b=df[df['TRADE_DT']==fund.iloc[i,1]]
        a=a.append(b)
    rm=pd.DataFrame(a['R'])
    rm['free']=0.0035 / 250#无风险利率每日
    rm['b1']=rm['R']-rm['free']
    rm['b2']=rm['b1']*rm['b1']
    X=np.array(rm.loc[:,'b1':'b2'])
    
    c=pd.DataFrame() 
    for i in range(a.shape[0]):
        m=fund[fund['PRICE_DATE']==a.iloc[i,1]]
        c=c.append(m)    
    ri=pd.DataFrame(c['RF'])
    ri['free']=0.0035 / 250
    ri['y']=ri['RF']-ri['free']
    Y=np.array(ri['y'])
    regr = linear_model.LinearRegression()
    regr.fit(X,Y)
    coef=regr.coef_
    inter=regr.intercept_ 
    name=fund.iloc[0,0]
    return name,inter,coef[1]

def order_reg(total,name,conn,df):
    reg2=[]
    for i in range(total):
        f=ind_fund(name[i],conn)
        reg1=reg(f,df)   
        reg2.append(reg1)    
    reg2=pd.DataFrame(reg2)
    reg2.index=reg2[0]
    reg2=reg2.drop(0,axis=1)
    beta=reg2[1].sort_values(ascending=False)
    alpha=reg2[2].sort_values(ascending=False)
    return alpha,beta

def score_reg(reg1,total):
    count=0
    for i in range(total):
        if reg1.iloc[i]>0:
            count=count+1
    s=math.ceil(count/4) 
    for i in range(total):
        if i<s:
            reg1.iloc[i]=5
        elif i>=s and i<2*s:
            reg1.iloc[i]=4
        elif i>=2*s and i<3*s:
            reg1.iloc[i]=3
        elif i>=3*s and i<count:
            reg1.iloc[i]=2
        else:
            reg1.iloc[i]=1
    return reg1

#结果输出
def out(s1,s2,s3,s4,s5,total):
    s=[]
    s.append(s1)
    s.append(s2)
    s.append(s3)
    s.append(s4)
    s.append(s5)
    s=pd.DataFrame(s)
    s.loc['total']=0
    for i in range (total):
        s.loc['total',s.columns[i]]=s.iloc[:4,i].mean()    
    s.index=['进攻','防御','守成','alpha','beta','total']
    s=s.transpose().sort_values(by ='total',ascending=False)
    file_name='score_48.xlsx'
    p=pd.ExcelWriter(file_name)
    s.to_excel(p,sheet_name="综合")
    p.save()    

#运行
if __name__=='_main_':
    conn = connect_database()
    df = index(conn)
    date=df['TRADE_DT']
    name=input_name('name.xlsx')
    total=name.shape[0] 
    z=[0,65,82,108,126,136,144,153,184,218,761,827,998,1040]
    d=[109,125,137,143,154,160,196,202,240,263,828,977,1041,1185]
    f=[66,81,161,183,219,239,264,760,978,997,1079,1168]
    score1=order_score(total,name,z,date,conn)
    score2=order_score(total,name,d,date,conn)
    score3=order_score(total,name,f,date,conn)
    line_reg=order_reg(total,name,conn,df)
    alpha=line_reg[0]
    beta=line_reg[1]
    score4=score_reg(alpha,total)
    score5=score_reg(beta,total)
    out(score1,score2,score3,score4,score5,total)


