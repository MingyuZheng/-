# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:54:58 2021

@author: zheng
"""

import jieba
from jieba.analyse import extract_tags
import pandas as pd 
import wordcloud
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import xlrd
import collections
import re
#coding=utf-8
#encoding=utf8

#处理数据，分离意见文本
#dataout=dataout.drop([141],axis=0)
#dataout['人次'].drop_duplicates()
#dataout[dataout['人次']>1].groupby(['人次']).count()['意见原述']
#因为有意见的频率影响，为了正确统计频词，需要把文本扩充为原来大小
def getdata(path,n,sheetname):
    dataout=pd.read_excel(path,sheetname=sheetname[n])
    dataout=dataout[dataout.columns[0:4]]
    for i in range(dataout.shape[0]):
        if dataout['人次'][i]>1:
            f=dataout['人次'][i]
            j=1
            while j < f:
                dataout=dataout.append(dataout.iloc[i,:])
                j+=1             
    sugout=dataout['意见原述']
    strout=''
    for i in sugout:
        strout += str(i)
    return strout
    
# jieba开始处理和分词
#基本函数
#cut generate generator  inputs must be  string
#lcut generate list inputs must be string
#将意见编程一个字符串
#jieba 的使用方法
#for w in jieba.cut("我爱您，Python！"):
#    print(w) 

# 读取过滤词表,停用词表  
def datacut(strout,remove_words):
    pattern = re.compile(u'\t|\n| |；|\.|。|：|：\.|-|:|\d|;|、|，|\)|\(|\?|"')
    strout=re.sub(pattern,'',strout)
    seg_list_exact = jieba.cut(strout, cut_all=False)  # 精确模式分词
#object_list  = list(seg_list_exact)    #降分词产生的迭代器转换为列表
    object_list = []
# 循环读出每个分词
    for word in seg_list_exact:
    #看每个分词是否在常用词表中或结果是否为空或\xa0不间断空白符，如果不是再追加
        if word not in remove_words and word != ' ' and word != '\xa0':
            object_list.append(word)  # 分词追加到列表
    return object_list
        

#做词频统计，制作词云
def worddict(object_list,n):
    word_counts = collections.Counter(object_list)  # 对分词做词频统计
    word_counts_top = word_counts.most_common(n)
    worddict={}
    for i in range(n):
        worddict[word_counts_top[i][0]]=word_counts_top[i][1]
    return worddict

#作图，产生词云
def wordcloudfig(worddict):
    font_path='C:/Windows/Fonts/simhei.ttf'
    tupath='toyuan.jpg'    
    tuoyuan= np.array(Image.open(tupath))   
    wcloud = WordCloud(font_path=font_path, width=800, height=600, mask=tuoyuan,mode='RGBA', background_color=None)
    wc=wcloud.generate_from_frequencies(worddict) #根据词频，必须是字典形式
    plt.axis('off')
    plt.imshow(wc)
    
#执行
path='suggestion.xlsx'
wb=xlrd.open_workbook(path)
sheetname=wb.sheet_names()
n=3 #出院
n=0 #门诊
n=2 #住院
strout=getdata(path,2,sheetname)
with open('removeword.txt', 'r', encoding="utf-8") as fp:
    remove_words = fp.read().split()    
object_list=datacut(strout,remove_words)
wddict=worddict(object_list,50)
wordcloudfig(wddict)
# wc.to_file('chuyuan.png')











































