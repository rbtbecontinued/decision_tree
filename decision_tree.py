#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 19:08:31 2017

@author: nanzheng
"""

import numpy as np
from metrics import *


class decision_tree_node():
    """
    决策树的一个节点。
    """
    
    def __init__(self, method="id3"):
        # 决策树生成算法
        # {"id3", "c4dot5"}
        self.method = method
        # 当前节点的类型
        # -1: 初始值
        # 0: 根节点
        # > 0: 枝节点或叶节点，数值代表节点的深度
        self.type = -1
        # 从当前节点划分出的各分支节点
        self.branches = []
        # 当前节点的父节点
        self.parent = None
        # 当前节点所在决策树的根节点
        self.root = None
        # 在根节点上保留各特征的值域
        self.range_of_features = []
        # 当前节点可选特征的索引
        self.candidate_features = []
        # 当前节点选择进行划分的特征的索引
        self.selected_feature = -1
        # 划分到当前节点的样本的索引
        self.index_of_samples = []
        # 当前节点（为叶节点时）的决策
        self.decision = None
        # 对新样本做出的预测
        self.prediction = None
        
        
    def is_root(self):
        """
        判断当前节点是否为根节点。
        
        :return (bool):
        """
        
        if self.type == 0:
            return True
        elif self.type > 0:
            return False
        
        
    def is_leaf(self):
        """
        判断当前节点是否为叶节点。
        
        :return (bool):
        """
        
        if (self.type > 0) and (len(self.branches) == 0):
            return True
        else:
            return False
        
        
    def is_branch(self):
        """
        判断当前节点是否为枝节点。
        
        :return (bool):
        """
        
        if (self.type > 0) and (len(self.branches) > 0):
            return True
        else:
            return False
        
        
    def set_root(self, root):
        """
        设定当前节点所在决策树的根节点。
        
        :param root: 当前决策树的根节点
        """
        
        self.root = root
        
        return None
        
        
    def set_depth(self, depth=0):
        """
        设定当前节点的深度。
        
        :param depth: 节点的深度
        """
        
        self.type = depth
        
        return None
    
    
    def set_parent(self, parent):
        """
        设定当前节点的父节点。
        
        :param parent: 父节点
        """
        
        self.parent = parent
        
        return None
    
    
    def _stop_growing(self, Y):
        """
        检查是否需在当前节点上继续划分。
        
        :param Y: 1d array，各样本对应的类别标签
        :return (bool):
        """
        
        # 当前节点正例和反例的个数
        n_pos, n_neg = 0, 0
        
        for i, i_sample in enumerate(self.index_of_samples):
            if Y[i_sample] == 1:
                n_pos += 1
            elif Y[i_sample] == -1:
                n_neg += 1
                
        # 当前节点全为正例
        if n_pos > 0 and n_neg == 0:
            self.decision = 1
            
            return True
        
        # 当前节点全为反例
        if n_pos == 0 and n_neg > 0:
            self.decision = -1
            
            return True
        
        # 当前节点可选特征个数为0
        if len(self.candidate_features) == 0:
            if n_pos + n_neg > 0:
                if n_pos >= n_neg:
                    self.decision = 1
                else:
                    self.decision = -1
                    
                return True
            
        # 当前节点样本个数为0
        if n_pos == 0 and n_neg == 0:
            # FIXME: 不能给出决策
            
            return True
        
        return False
    
    
    def _select_feature(self, X, Y, range_of_features, selected_samples=None):
        """
        从当前节点可选特征中选择使得信息增益最大的特征。
        
        :param X: N * M array，数据集，N为样本个数，M为特征个数
        :param Y: 1d array，各样本对应的类别标签，{-1, 1}
        :param range_of_features: list，各特征的值域
        :param selected_samples: list，当前节点样本的索引
        :return i_selected_feature: 选定的特征的索引
        :return samples_devided: list，各元素也为list，记录利用选定的特征划分后，
            进入各分支的样本的索引
        """
        
        # 利用各特征进行划分获得的信息增益
        delta_I = np.empty(len(self.candidate_features))
        # 利用各特征进行划分后，进入各分支的样本的索引
        i_devided_samples = [[] for i in range(len(self.candidate_features))]
        for i in range(len(self.candidate_features)):
            # 计算利用当前特征进行划分获得的信息增益
            delta_I[i], i_devided_samples[i] = metrics.info_gain( \
                   self.candidate_features[i], X, Y, 
                   range_of_features[self.candidate_features[i]], 
                   selected_samples)
            
        # 当前选用ID3方法
        if self.method is "id3":
            # 选择获得信息增益最大的特征
            i_selected_feature = self.candidate_features[np.argmax(delta_I)]
            samples_devided = i_devided_samples[np.argmax(delta_I)]
            
        # 当前选用C4.5方法
        elif self.method is "c4dot5":
            # 利用各特征进行划分获得的信息增益率
            delta_I_v = np.zeros(len(self.candidate_features))
            # 平均信息增益
            mean_delta_I = np.mean(delta_I)
            
            for i in range(len(self.candidate_features)):
                # 从信息增益高于平均水平的特征中，选择使得信息增益率最高的特征
                if delta_I[i] >= mean_delta_I:
                    # 计算利用当前特征进行划分获得的信息增益率
                    delta_I_v[i], i_devided_samples[i] = metrics.gain_ratio( \
                             self.candidate_features[i], X, Y, 
                             range_of_features[self.candidate_features[i]], 
                             selected_samples)
                    
            i_selected_feature = self.candidate_features[np.argmax(delta_I_v)]
            samples_devided = i_devided_samples[np.argmax(delta_I_v)]
        
        return i_selected_feature, samples_devided
    
    
    def train(self, X, Y, range_of_features, 
              index_of_samples=None, candidate_features=None):
        """
        从当前节点继续进行训练。
        
        :param X: N * M array，数据集，N为样本个数，M为特征个数
        :param Y: 1d array，各样本对应的类别标签，{-1, 1}
        :param range_of_features: list，各特征的值域
        :param index_of_samples: list，当前节点样本的索引
        :param candidate_features: list，当前节点可选特征的索引
        """
        
        # 将当前节点作为根节点
        if self.type == -1:
            self.type = 0
            self.root = self
            self.range_of_features = range_of_features
        
        # 划分到当前节点的样本的索引
        if index_of_samples is None:
            index_of_samples = [i for i in range(Y.shape[0])]
        self.index_of_samples = index_of_samples
        
        # 当前节点可选特征的索引
        if candidate_features is None:
            candidate_features = [i for i in range(len(range_of_features))]
        self.candidate_features = candidate_features
        
        # 检查是否需要对当前节点继续划分
        if self._stop_growing(Y):
            return None
        
        # 选择使得信息增益最大的特征
        self.selected_feature, samples_devided = \
            self._select_feature(X, Y, range_of_features, index_of_samples)
            
        # 当前节点的子节点的可选特征的索引
        candidate_features_for_child_node = []
        for i, i_feature in enumerate(self.candidate_features):
            if i_feature != self.selected_feature:
                candidate_features_for_child_node.append(i_feature)
            elif i_feature == self.selected_feature:
                continue
            
        # 利用所选的特征从当前节点继续进行划分
        for i in range(range_of_features[self.selected_feature].shape[0]):
            # 实例化新的子节点
            new_child_node = decision_tree_node(method=self.method)
            # 当前节点为每个子节点的父节点
            new_child_node.set_parent(self)
            # 为子节点设置根节点
            new_child_node.set_root(self.root)
            # 每个子节点为当前节点的分支
            self.branches.append(new_child_node)
            # 设定子节点的深度
            new_child_node.set_depth(self.type + 1)
            # 继续训练子节点
            new_child_node.train(X, Y, range_of_features, samples_devided[i], 
                                 candidate_features_for_child_node)
        
        return None
    
    
    def predict(self, x):
        """
        对当前输入样本的类别进行预测。
        
        :param x: 1d array，预测样本
        :return self.prediction: 预测结果，由根节点返回
        """
        
        # 当前节点为叶节点，将决策赋给根节点
        if self.is_leaf():
            self.root.prediction = self.decision
            
            return None
        
        # 当前节点不是叶节点
        else:
            # 根据当前节点所选的特征，进入对应的枝节点继续进行决策
            for i in range( \
                    self.root.range_of_features[self.selected_feature].shape[0]):
                if x[self.selected_feature] == \
                        self.root.range_of_features[self.selected_feature][i]:
                    self.branches[i].predict(x)
                    
                    break
                
        # 由根节点返回预测结果
        if self.is_root():
            return self.prediction
        
        return None
    
    
    def traverse(self, index=0):
        """
        遍历决策树。
        
        :param index: 当前节点在其父节点所有分支中的索引
        """
        
        print("\t" * self.type, end="")
        print("--------------------")
        print("\t" * self.type, end="(")
        print(self.type, end=", ")
        print(index, end=") ")
        
        if self.is_root():
            print("Root Node", end="\n\n")
            print("\t" * self.type, end="")
            print("Selected Feature: ", end="")
            print(self.selected_feature)
            
        elif self.is_branch():
            print("Branch Node", end="\n\n")
            print("\t" * self.type, end="")
            print("Selected Feature: ", end="")
            print(self.selected_feature)
            
        elif self.is_leaf():
            print("Leaf Node", end="\n\n")
            print("\t" * self.type, end="")
            print("Decision: ", end="")
            print(self.decision)
        
        print("\t" * self.type, end="")
        print("^^^^^^^^^^^^^^^^^^^^")
        
        for i in range(len(self.branches)):
            self.branches[i].traverse(i)
            
        return None
    
    