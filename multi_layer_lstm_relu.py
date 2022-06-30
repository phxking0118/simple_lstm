# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:37:04 2021

@author: Administrator
"""

##define root path
##只要修改好root path和内存空间就都能跑
##定义相关激活函数等从setting里直接import 这也主文件就简洁多了
## setting
import numpy as np   
import pandas as pd
from datetime import datetime
import warnings
from lstm_setting import *
import os
import matplotlib as plt
warnings.filterwarnings('ignore')





#%%%
#导入网络的输入数据以及目标矩阵
path_list = ['aim_dict\\', 'bias\\', 'bias_y\\', 'c_0\\', 'candidate_dict\\', 'delta_candidate_dict\\'
    , 'delta_end_dict\\', 'delta_forget_dict\\', 'delta_input_dict\\', 'delta_output_dict\\'
    , 'factors_value\\', 'forget_dict\\', 'h_0\\', 'input_dict\\'
    , 'output_dict\\', 'start_dict\\', 'state_dict\\', 'test_lstm\\'
    , 'W\\', 'W_y\\','delta_layer_dict\\']
rootpath=r'D:\lstm\\'
## 创建文件路径来储存npy文件 避免报错
path_setting(rootpath, path_list)
#如pdf中所述，取8个时间作为时间上的维度
time_step=8
#无实际意义，只是用于下述迭代计算
lens=time_step*5-4
#训练集中每个时间点上含有1500天的全部股票，作为样本
days_span=1500
#定义输入层的输入矩阵,并进行存储
factor_made(rootpath,lens,days_span)
    
#定义目标输出矩阵，以收益率为标准进行排序，把每日收益率最好的前160只股票标为1，其余的标为0
#导入收益率数据
df=pd.read_csv(rootpath+r"Stock500\\RawRet.csv")
df=df.set_index('DATETIME')
#排序
df=df.rank(axis=1,ascending=False)  
#打标签
df[df<=160]=1    
df[df>160]=0 
#因为股票的收益率在两天后体现，向前移动两个单位   
df=df.shift(-2) 
df_matrix=np.array(df)   

#根据一字板对目标输出矩阵进行修正
#导入一字板数据
df_one=pd.read_csv(rootpath + r'Stock500\\yizi.csv')
df_one=df_one.set_index('DATETIME')
#因为股票是在后一天进行交易，向前移动一个单位
df_one=df_one.shift(-1)
#将为1的值柏标为0，其余标为1
df_one[df_one != 1]=2
df_one[df_one == 1]=0
df_one[df_one==2]=1

#得到修正后的目标输出矩阵
aim_matrix=df_matrix*np.array(df_one)
#对目标输出矩阵，按照输入矩阵的方法提取时间序列
for i in range(0,lens,5):
    aim_time_list=aim_matrix[i:days_span+i,:]
    #展开成向量
    aim_time_list=aim_time_list.flatten().reshape(1,aim_time_list.size)
    #存储
    i=i/5
    np.save(rootpath+'aim_dict\\%d.npy'%i,aim_time_list)
    
#%%  
#删除nan，缩小矩阵维度
#导入一个矩阵，为初始矩阵确定行列
q = np.load(rootpath+'start_dict\\0_0.npy')
a = np.ones((q.shape[0],q.shape[1]))
for i in range(time_step):
    p=np.load(rootpath+'start_dict\\0_%d.npy'%i)
#依元素将所有矩阵相乘
    a = a*p
#提取相乘结果矩阵中全不为nan的列的标签
nan_index_st = ~np.any(np.isnan(a),axis=0)
a = np.ones((q.shape[0],q.shape[1]))
for i in range(time_step):
    p=np.load(rootpath+'aim_dict\\%d.npy'%i)
#同上，找到每一time_step都不为0的标签
    a = a*p
#对标签取“且”得到总标签
nan_index_ai = ~np.any(np.isnan(a),axis=0)
total_index = ~(nan_index_ai* nan_index_st)    

p={}
for i in range(time_step):
    p=np.load(rootpath+'start_dict\\0_%d.npy'%i)
    #根据找到的index删除nan
    p=p[:,~total_index]
    np.save(rootpath+'start_dict\\0_%d.npy'%i,p)

for i in range(time_step):
    p=np.load(rootpath+'aim_dict\\%d.npy'%i)
    p=p[:,~total_index]
    np.save(rootpath+'aim_dict\\%d.npy'%i,p)

#%%%
#使用训练集进行训练
#初始化参数
#因子数量
factor_size=182
#输出神经元数量
aim_size=1
#学习率
learning_rate = 0.01/time_step
#阈值，确定是否收敛
beta=1
#lstm的层数及每层的神经元个数列表
number=[10,5]
#时间序列长度
time_len=8
#W为权值矩阵字典，第一个参数表示第几次循环，每次循环对应一个字典，即第二个参数等于隐藏层的个数，每个隐藏层又对应一个字典
#包括8个权值矩阵，从0到7分别对应W_fh,W_fx,W_ih,W_ix.W_ch,W_cx,W_oh,W_ox
#bias为偏差项字典，第一个参数表示第几次循环，每次循环对应一个字典，即第二个参数等于隐藏层的个数，每个隐藏层又对应一个字典
#bias字典中对于的4个向量分别表示forget,input,candidate,output的偏度
W={}   
bias={}
#循环计数器
j=1  
#lstm层数
layer_len=len(number)         
for i in range(1000):         #1000只是为了取一个较大的循环次数值，无实际含义
    W[i]={}
    bias[i]={}
    for k in range(layer_len):
        W[i][k]={}
    for k in range(layer_len):
        bias[i][k]={}
#之后W共有三个指标，分别表示循环次数，lstm的第几层，每次循环中每层对应的8个W

#损失值
loss=[]
#初始化损失值向量
loss.append(0)
loss.append(beta+1)

#对每一层中的权值矩阵定义初值，一层中共有8个权值矩阵
#list_1中的键对应的矩阵为方阵
#list_2中的键对应的矩阵行数与列数与上一层和本层神经元个数有关
#两个列表中的初始化矩阵分别进行定义
list_1=[0,2,4,6]
list_2=[1,3,5,7]
for i in list_1:
    for k in range(0,layer_len):
        #初始化方法采用标准正态分布数据，并稍加修改
        W[j][k][i]=np.random.randn(number[k],number[k])/(np.sqrt(number[k]))

for i in list_2:
    W[j][0][i]=np.random.randn(number[0],factor_size)/(np.sqrt(factor_size))
    for k in range(1,layer_len):
        W[j][k][i]=np.random.randn(number[k],number[k-1])/(np.sqrt(number[k-1]))

#对偏差项进行定义        
for i in range(4):
    for k in range(layer_len):
        bias[j][k][i]=abs(np.random.randn(number[k],1)/100)
    
    
#最后一个隐藏层到输出层的权值矩阵、偏差项及其初始化
W_y={}
bias_y={}
#初始化方法同上述定义
W_y[j]=np.random.randn(aim_size,number[layer_len-1])/(np.sqrt(number[layer_len-1]))
''''''
bias_y[j]=abs(np.random.randn(aim_size,1)/100)

#对初始化的c_0和h_0进行初始化
'''?'''
#sub_c_0为一个字典，字典中的键对应一个隐藏层，对应的元素为一个向量，表示对于一个样本计算时的初始细胞状态，所以样本共用一组数据
#sub_h_0为一个字典，字典中的键对应一个隐藏层，对应的元素为一个向量，表示对于一个样本计算时的初始h_0，所有样本共用1组数据
sub_c_0={}
sub_h_0={}
#利用sub_c_0,sub_h_0生成列数与样本数量相同的矩阵
c_0={}
h_0={}
#导入一个矩阵，其列数等于样本数量
start=np.load(rootpath+'start_dict\\0_0.npy')
for i in range(layer_len):
    #利用sub_c_0,sub_h_0生成列数与样本数量相同的矩阵
    sub_c_0[i]=np.random.randn(number[i],1)/(np.sqrt(start.shape[1]))
    c_0[i]=np.ones((len(sub_c_0[i]),start.shape[1]))
    '''c_0[i] = np.random.randn(number[i],start.shape[1])后面计算方法计算的的每一列都是一样的'''
    c_0[i]=sub_c_0[i]*c_0[i]
    sub_h_0[i]=np.random.randn(number[i],1)/(np.sqrt(start.shape[1]))
    h_0[i]=np.ones((len(sub_h_0[i]),start.shape[1]))
    h_0[i]=sub_h_0[i]*h_0[i]

del start
#以上为初始化参数
#c_0,h_0,W,W_y,bias,bias_y均不需要释放内存
#循环迭代
while (abs(loss[j]-loss[j-1])>beta):
    print('开始循环')
    print(datetime.now())
    #正向计算
    #对网络中出输出层外的其它网络层进行计算
    forward(rootpath,c_0,h_0,W[j],bias[j],time_len,layer_len)
    #对网络输出层进行计算
    forward_last_layer(rootpath,W_y[j],bias_y[j],time_len,layer_len)

    print('前向运算结束')
    #计算损失值,同时计算输出层残差
    loss_value=0
    for i in range(time_len):
        #导入输出层输出矩阵
        end=np.load(rootpath+'start_dict\\%d_%d.npy'%(layer_len+1,i))
        '''1*200w'''
        #导入定义好的目标矩阵
        aim=np.load(rootpath+'aim_dict\\%d.npy'%i)
        '''1*200w'''
        #计算损失值
        loss_value+=np.nansum((end-aim)**2)
        #计算输出层残差并存储
        delta_end=(end-aim)*drelu_vec(end)
        '''1*200w'''
        np.save(rootpath+'delta_end_dict\\%d.npy'%i,delta_end)
        #修改W_y与bias_y,
        '''最后一个隐藏层到输出层的权值矩阵'''
        delta=0
        delta_bias=0
        #导入输出层的输入矩阵
        start=np.load(rootpath+'start_dict\\%d_%d.npy'%(layer_len,i))
        '''5*200w'''
        #计算W_y的梯度
        delta+=dot_np(delta_end,start.T)
        '''1*5'''
        #计算bias_y的梯度
        delta_add=np.nansum(delta_end,axis=1)/delta_end.shape[1]
        '''这里delta_end.shape[1] = 1'''
        delta_bias+=delta_add.reshape((delta_add.size,1))
        '''.size是矩阵的元素个数，这里也是1 1*1'''
    #进行更新
    W_y[j+1]=W_y[j]-learning_rate*delta
    bias_y[j+1]=bias_y[j]-learning_rate*delta_bias
    loss.append(loss_value)
    #对损失值进行判断，如果损失值收敛，则跳出循环
    if (abs(loss[j+1]-loss[j])<=beta):
        break
    #更新循环计数器
    j+=1

    #计算残差并更新W

    W[j][layer_len-1],bias[j][layer_len-1]=backward_delta_last(rootpath,c_0[layer_len-1],
                                                               h_0[layer_len-1],W[j-1][layer_len-1],W_y[j-1],
                                                               bias[j-1][layer_len-1],time_len,layer_len,learning_rate)
    #下述使用unuse只是为了带入方便以及防止修改已有数据
    unuse_W_1={}
    unuse_bias_1={}
    unuse_c_0={}
    unuse_h_0={}
    for i in range(layer_len):
        unuse_W_1[i]=W[j-1][i].copy()

    for i in range(layer_len-1):
        unuse_c_0[i]=c_0[i]
        unuse_h_0[i]=h_0[i]
        unuse_bias_1[i]=bias[j-1][i].copy()
    unuse_W_2,unuse_bias_2=backward_delta(rootpath,unuse_c_0,unuse_h_0,unuse_W_1,unuse_bias_1,time_len,layer_len,learning_rate)
    for i in range(layer_len-1):
        W[j][i]=unuse_W_2[i].copy()
        bias[j][i]=unuse_bias_2[i].copy()


    #存储数据
    #以下6个数据对于一个网络而言较为重要，及时存储
    np.save(rootpath+'W\\%d.npy'%j,W[j])
    np.save(rootpath+'W_y\\%d.npy'%j,W_y[j])
    np.save(rootpath+'bias\\%d.npy'%j,bias[j])
    np.save(rootpath+'bias_y\\%d.npy'%j,bias_y[j])
    np.save(rootpath+'c_0\\%d.npy'%j,sub_c_0)
    np.save(rootpath+'h_0\\%d.npy'%j,sub_h_0)




    print(datetime.now())
    print(loss_value)
    
#计算仅用均值预测的loss_value
#for i in range(time_len):
#    aim = np.load(rootpath + 'aim_dict\\%d.npy'%i)
#    aim_mean= aim.mean()

