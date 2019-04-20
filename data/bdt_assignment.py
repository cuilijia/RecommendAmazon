# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 15:05:55 2019

@author: 钱秉诚
"""

import json
import random as rd
import numpy as np


def produce(produceFile,datasize):
    a = open("sports_outdoor_matrix.json", "r")
    b = open(produceFile, "w", encoding='utf-8')
    s=json.load(a)
    s["-1"]=[-1]

    #随机走路的function 返回一个走路的列表
    def get_random_walk(size,step_len,file):
        result=np.arange(size*step_len).reshape(size, step_len)
        for i in range(size):
            x=str(rd.randint(0,len(file)-1))
            for k in range(step_len):
                result[i][k]=x
                if x in file:
                    x=str(rd.sample(file[x],1)[0])
                else:
                    file[x]=[-1]
                    x=str(file[x][0])
        return result

    dict1={}

    result1=get_random_walk(datasize,10,s)
    #将结果从numpy矩阵变成列表
    result2=result1.tolist()
    #将数字变成字符
    def tostr(list_int):
        for i in range(len(list_int)):
            for k in range(len(list_int[i])):
                list_int[i][k]=str(list_int[i][k])
        return list_int
    #result2表示字符格式的列表
    result2=tostr(result2)
    #生成一个json文件
    for num,element in enumerate(result2):
        dict1[num]=element

    data1=[dict1]
    json.dump(data1,b,ensure_ascii=False)
    a.close()

for i in range(1,11):
    produceFile="data"+str(i)+".json"
    produce(produceFile,10000)
