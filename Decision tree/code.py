#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
from math import log
import operator

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import statistics
import os
import numpy as np
from numpy import *
import sys
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

from collections import defaultdict
import math

class Tree:
    def __init__(self,value=None, trueBranch=None, falseBranch=None, results=None, col=-1, data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.data = data

def load_data(data_path):
    current_dir = os.path.dirname(os.path.abspath("__file__"))
    path_and_file = os.path.join(current_dir, data_path)
    dataset=pd.read_csv(path_and_file,delimiter="\t", header=None)
    dataset = dataset.values
    return dataset    

def classCount(data):
    count_dict = {}
    #print("data: ", data, type(data))
    classCol = [row[-1] for row in data]
    for item in classCol:
        
        if item in count_dict.keys():
            count_dict[item] +=1
        else:
            count_dict[item] = 1
    return count_dict


    
# gini()
def gini(rows):
    counts = classCount(rows)
    #print("Counts is a {}, and contains: {}".format(type(counts), counts))
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2

    return impurity

def calc_info_gain(trueRows, falseRows, parent_gini):
    p = float(len(trueRows)) / (len(trueRows) + len(falseRows))
    return parent_gini - (p * gini(trueRows) + (1 - p) * gini(falseRows)) # recall gini for left and right children


def splitDatas(rows, value, column):
    left = []
    right = []

    if isinstance(value, int) or isinstance(value, float):
        for row in rows:
            if row[column] >= value:
                left.append(row)
            else:
                right.append(row)
    else:
        for row in rows:
            if row[column] == value:
                left.append(row)
            else:
                right.append(row)
    return left, right


def buildDecisionTree(rows, layer, node, evaluationFunction=gini):
    
    """
    currentGain: the whole dataset gini
    """
    currentGain = evaluationFunction(rows)
    
    n_features = len(rows[0]) - 1
    
    rows_length = len(rows)
    
    best_gain = 0.0
    best_value = None
    best_set = None
    
    # choose the best gain
    for col in range(n_features):
        unique_val_list_inCol = set([x[col] for x in rows])
        for value in unique_val_list_inCol:
            left, right = splitDatas(rows, value, col)
            """
            calculate gini
            """
            gain = calc_info_gain(left, right, currentGain)
            if gain > best_gain:
                best_gain = gain
                best_value = (col, value)
                best_set = (left, right)
    
    if(printTree):
        print("layer: {},node:{}, best_value:{}".format(layer, node, best_value))
    
    layer += 1
    
    #
    # stop or not stop

    if best_gain > 0:
        trueBranch = buildDecisionTree(best_set[0],layer,"left", evaluationFunction)
        falseBranch = buildDecisionTree(best_set[1],layer,"right", evaluationFunction)
        return Tree(col=best_value[0], value = best_value[1], trueBranch = trueBranch, falseBranch=falseBranch)
    else:
        return Tree(results=classCount(rows), data=rows)


def classify(data, tree):
    if tree.results != None:
        return tree.results
    else:
        branch = None
        v = data[tree.col]
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        else:
            if v == tree.value:
                branch = tree.trueBranch
            else:
                branch = tree.falseBranch
        return classify(data, branch)


def evaluate(predictTruth, groundTruth):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(groundTruth)):
        for j in range(len(predictTruth)):
            if int(groundTruth[i]) == 1 and int(predictTruth[i]) == 1:
                tp+=1
            if int(groundTruth[i]) == 0 and int(predictTruth[i]) == 1:
                fp+=1
            if int(groundTruth[i]) == 1 and int(predictTruth[i]) == 0:
                fn+=1
            if int(groundTruth[i]) == 0 and int(predictTruth[i]) == 0:
                tn+=1
    
    #P : PRECISION
    # R : RECALL
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    try:
        P = (tp)/(tp+fp)
    except:
        P = 0
        
    try:
        R = (tp)/(tp+fn)
    except:
        R = 0
        
    try:
        F = 2*(P*R)/(P+R)
    except:
        F = 0
        
    return accuracy, P, R, F


def evaluate_std(predictTruth, groundTruth):
    tn, fp, fn, tp = confusion_matrix(groundTruth, predicTruth).ravel()
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    P = (tp)/(tp+fp)
    R = (tp)/(tp+fn)
    F = 2*(P*R)/(P+R)
    return accuracy, P, R, F


if __name__ == '__main__':
    
    printTree = False
    N = 10
    A = []
    P = []
    R = []
    F = []
    
    kf = KFold(n_splits = N)
    trainFile = "../project3_dataset4.txt"
    testFile = ""
    
    ac = []
    if testFile == "":
        dataset = load_data(trainFile)
        for train_index, test_index in kf.split(dataset):
            layer = 0
            node = "root"
            dataSet= dataset[train_index]
            test = dataset[test_index]
            labels = [x for x in range(len(dataSet[0])-1)]
            labels_tmp = labels[:]
            dataSet = dataSet.tolist()
            decisionTree = buildDecisionTree(dataSet, layer,node,evaluationFunction=gini)
#             print("--------------------------------------------------------------------")
#             print(decisionTree)
#             print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            groundTruth = test[:, -1]
            test = test[:, :-1]
            testSet = test.tolist()
            predicTruth = []
            for i in range(len(testSet)):
                each_dict = classify(testSet[i], decisionTree)
                predicTruth.append(list(each_dict.keys())[0])

            ac, p, r, f = evaluate(predicTruth,groundTruth)
#             ac, p, r, f = evaluate_std(predicTruth, groundTruth)

            A.append(ac)
            P.append(p)
            R.append(r)
            F.append(f)
        A_mean = sum(A)/len(A)
        P_mean = sum(P)/len(P)
        R_mean = sum(R)/len(R)
        F_mean = sum(F)/len(F)

        print("Accuracy = {}".format(A_mean))
        print("Precision = {}".format(P_mean))
        print("Recall = {}".format(R_mean))
        print("F measure = {}".format(F_mean))
    else:
        test = load_data(testFile)
        dataSet= load_data(trainFile)
        labels = [x for x in range(len(dataSet[0])-1)]
        labels_tmp = labels[:]
        dataSet = dataSet.tolist()
        layer = 0
        node = "root"
        decisionTree = buildDecisionTree(dataSet,layer,node, evaluationFunction=gini)
        groundTruth = test[:, -1]
        test = test[:, :-1]
        testSet = test.tolist()

        predicTruth = []
        for i in range(len(testSet)):
            each_dict = classify(testSet[i], decisionTree)
            predicTruth.append(list(each_dict.keys())[0])

        ac, p, r, f = evaluate(predicTruth,groundTruth)
#         ac, p, r, f = evaluate_std(predicTruth, groundTruth)
        print("Accuracy = {}".format(ac))
        print("Precision = {}".format(p))
        print("Recall = {}".format(r))
        print("F measure = {}".format(f))
        
         


# In[ ]:





# In[ ]:




