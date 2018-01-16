#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 18:28:08 2017

@author: nanzheng
"""

import numpy as np


class metrics():
    """
    不纯度的度量。
    """
    
    @staticmethod
    def entropy(P):
        """
        计算信息熵。
        $$ I = -\sum_{i = 1}^K P_i \log_2{P_i} $$
        
        :param P: 1 * K array，各事件发生的概率
        :rerturn I: 信息熵
        """
        
        I = 0
        for i in range(P.shape[0]):
            if P[i] == 0:
                continue
            
            I -= P[i] * np.log2(P[i])
            
        return I
    
    
    @staticmethod
    def gini(P):
        """
        计算Gini指数。
        $$ I = 1 - \sum_{i = 1}^K P_i^2 $$
        
        :param P: 1 * K array，各事件发生的概率
        :rerturn I: Gini指数
        """
        
        I = 1
        for i in range(P.shape[0]):
            I -= P[i]**2
            
        return I
    
    
    @classmethod
    def info_gain(cls, i_feature, X, Y, range_of_feature, 
                  selected_samples=None, metric=None):
        """
        计算信息增益。
        $$ \text{Gain}(D) = I(D) - \sum_{i = 1}^K \frac{|D^i|}{|D|} I(D^k) $$
        
        :param i_feature: 当前选择的特征的索引
        :param X: N * M array，数据集，N为样本个数，M为特征个数
        :param Y: 1d array，各样本对应的类别标签，{-1, 1}
        :param range_of_feature: 1d array，当前选择的特征的值域
        :param selected_samples: list，划分到当前节点的各样本的索引
        :param metric: func，不纯度的度量，{cls.entropy, cls.gini}
        :return delta_I: 利用当前选择的特征进行划分获得的信息增益
        :return i_samples_devided: list，每个元素记录划分后，当前选择的特征的一个取值对
            应的各样本的索引
        """
        
        if metric is None:
            metric = cls.entropy
        
        if selected_samples is None:
            selected_samples = [i for i in range(Y.shape[0])]
            
        # 划分前正例和反例的概率
        P_0 = np.zeros(2)
        # 划分后当前特征每个取值对应的样本中，正例和反例的概率
        P = np.zeros([range_of_feature.shape[0], 2])
        # 划分后当前特征每个取值对应的样本的索引
        i_samples_devided = [[] for i in range(range_of_feature.shape[0])]
        for i in range(len(selected_samples)):
            if Y[selected_samples[i]] == 1:
                P_0[0] += 1
                
                for j in range(range_of_feature.shape[0]):
                    if X[selected_samples[i]][i_feature] == range_of_feature[j]:
                        P[j][0] += 1
                        i_samples_devided[j].append(selected_samples[i])
                        
                        break
            elif Y[selected_samples[i]] == -1:
                P_0[1] += 1
                
                for j in range(range_of_feature.shape[0]):
                    if X[selected_samples[i]][i_feature] == range_of_feature[j]:
                        P[j][1] += 1
                        i_samples_devided[j].append(selected_samples[i])
                        
                        break
                    
        P_0 /= len(selected_samples)
        # 当前节点划分之前的不纯度
        I_0 = metric(P_0)
        
        I = np.empty(range_of_feature.shape[0])
        w = np.empty(range_of_feature.shape[0])
        for i in range(range_of_feature.shape[0]):
            # 当前特征的当前取值对应的样本个数 / 样本总个数
            w[i] = np.sum(P[i, :]) / len(selected_samples)
            # 划分后当前特征的当前取值对应的样本中，正例和反例的概率
            P[i, :] /= np.sum(P[i, :])
            # 划分后当前特征的当前取值对应的不纯度
            I[i] = metric(P[i, :].reshape(-1))
            
        # 信息增益
        delta_I = I_0 - np.dot(w, I)
        
        return delta_I, i_samples_devided
    
    
    @classmethod
    def gain_ratio(cls, i_feature, X, Y, range_of_feature, 
                   selected_samples=None, metric=None):
        """
        计算信息增益率。
        
        :param i_feature: 当前选择的特征的索引
        :param X: N * M array，数据集，N为样本个数，M为特征个数
        :param Y: 1d array，各样本对应的类别标签，{-1, 1}
        :param range_of_feature: 1d array，当前选择的特征的值域
        :param selected_samples: list，划分到当前节点的各样本的索引
        :param metric: func，不纯度的度量，{cls.entropy, cls.gini}
        :return delta_I_r: 利用当前选择的特征进行划分获得的信息增益率
        :return i_samples_devided: list，每个元素记录划分后，当前选择的特征的一个取值对
            应的各样本的索引
        """
        
        # 计算信息增益
        delta_I, i_samples_devided = cls.info_gain(i_feature, X, Y, 
                                                   range_of_feature, 
                                                   selected_samples, metric)
        
        # 计算分裂信息
        I_v = np.array([len(i_samples_devided[i]) \
                        for i in range(len(i_samples_devided))])
        I_v = I_v / np.sum(I_v)
        I_v = -np.dot(I_v, np.log2(I_v))
        
        # 计算信息增益率
        delta_I_r = delta_I / I_v
        
        return delta_I_r, i_samples_devided
    
    