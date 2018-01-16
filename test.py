#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 23:21:18 2017

@author: nanzheng
"""

import numpy as np
from decision_tree import *


X = np.array([
        [0, 0, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 2],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 2],
        [1, 0, 1],
        [0, 0, 2],
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0]])

Y = np.array([
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        1,
        -1])

range_of_features = [
        np.array([0, 1]),
        np.array([0, 1]),
        np.array([0, 1, 2])]


Decision_Tree = decision_tree_node()
Decision_Tree.train(X, Y, range_of_features)
Decision_Tree.traverse()


x = np.array([1, 1, 0])
y = Decision_Tree.predict(x)

