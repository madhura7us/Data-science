
# coding: utf-8

# In[1]:



# coding: utf-8

# In[2]:


import numpy as np
from math import log
import operator
# import treePlotter
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import statistics
import os
import numpy as np
from numpy import *
import os
import numpy as np
from numpy import *
from sklearn.model_selection import KFold
import sys
from heapq import heappush, heappop
import numpy as np
from numpy import *
import pandas as pd
from scipy.spatial import distance
import heapq
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import *

from collections import defaultdict
import math



# In[2]:


data = pd.read_csv("project3_dataset2.txt", sep='\t',header=None)
vector = data.as_matrix()


# In[3]:





class Tree:
    def __init__(self,debug=None, value=None, trueBranch=None, falseBranch=None, results=None, col=-1, summary=None, data=None):
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results
        self.col = col
        self.summary = summary
        self.data = data
        self.debug = debug

    def __str__(self):
        print("index: {}, value: {}".format(self.col, self.value))
        print("left child: ", self.trueBranch)
        print("right child: ", self.falseBranch)
        print("dict of classes Counts: ",self.results)
        print("debug: ", self.debug)
        print(self.summary)
        return ""

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
    # 计算gini的值(Calculate GINI)
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
    # 根据条件分离数据集(splitDatas by value, column)
    # return 2 part（list1, list2）

    list1 = []
    list2 = []

    if isinstance(value, int) or isinstance(value, float):
        for row in rows:
            if row[column] >= value:
                list1.append(row)
            else:
                list2.append(row)
    else:
        for row in rows:
            if row[column] == value:
                list1.append(row)
            else:
                list2.append(row)
    return list1, list2



def buildDecisionTree(rows, evaluationFunction=gini):
    
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
    dcY = {'impurity': '%.3f' % currentGain, 'training dataset size': '%d' % rows_length}
    #
    # stop or not stop

    if best_gain > 0:
        trueBranch = buildDecisionTree(best_set[0], evaluationFunction)
        falseBranch = buildDecisionTree(best_set[1], evaluationFunction)
        return Tree(col=best_value[0], value = best_value[1], trueBranch = trueBranch, falseBranch=falseBranch, summary=dcY)
    else:
        return Tree(debug = "best_gain <= 0", results=classCount(rows), summary=dcY, data=rows)


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
    
    # P : PRECISION
    # R : RECALL
    print(tp, fp , tn ,fn)
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    P = (tp)/(tp+fp)
    R = (tp)/(tp+fn)
    F = 2*(P*R)/(P+R)
    
    
    return accuracy,P,R,F


# In[4]:


data.columns[-1]


# In[5]:


import random
k = 10
split = 5
kf = KFold(n_splits=k)

#  all columns
M = data.columns
# converted to list
Mpsu = M.tolist()
Mpsu.remove(Mpsu[-1])
# declare number of attribtues required
m = 5
# making list of attributes choosen randomly
mList = []
for x in range(k):
    random.shuffle(Mpsu)
    psuX = Mpsu[0:m]
    psuX.append(data.columns[-1])
    mList.append(psuX)
print(" M LIST " , mList)
# j=0


# In[6]:


a = 0
pre = 0
re = 0
fM = 0
for train , test in kf.split(vector):    

    newData = data
    
    trainResult = np.array_split(train, split)
    
    testResult = np.array(test)
    
    test = vector[testResult]
    testList = []
    for x in range(len(mList)):
        testList.append(test[:,mList[x]])
    
    
    groundTruth = test[:,-1]
    decisionTreeLst = []
#     // TREE FORMATION
    
    for tr in range(len(trainResult)):
#         m = 
        dataSet = vector[trainResult[tr]]
        dataSet = dataSet[:,mList[tr]]
        dataSet = dataSet[:,]
        decisionTreeLst.append(buildDecisionTree(dataSet , evaluationFunction=gini))
#     // decisicionTreeLst contains 5 trees 

    refineTestList = []
    for x in testList:
        refineTest = x[:,:-1] 
        refineTest = refineTest.tolist()
        refineTestList.append(refineTest)
    
    # refineTEST m1 - m5
    # decisionTree   m1 - m5
#     refineTest = test[:,:-1]
#     refineTest = refineTest.tolist()
    allResult = []
    
    import itertools
    for refineTest,decisionTree in zip(refineTestList , decisionTreeLst):
        result = []
        for i in range(len(refineTest)):
            eachDict = classify(refineTest[i],decisionTree)
            result.append(list(eachDict.keys())[0])
        allResult.append(result)
        
#     fin = []
    fin = [list(i) for i in zip(*allResult)]
                
    
#     
        
#     print(" AYUSH ", fin )
#     dT = decisicionTreeLst[1] # m0
#     tL = refineTestList[1]  # m0
    
#     result = []
#     for t in tL:
#         eachDict = classify(t,dT)
#         result.append(list(eachDict.keys())[0])
#     print(" result " , result)
        
#     print(refineTestList)
# #   print(allResult)
# #   MAJORITY VOTING
# #     print(allResult)
    majorityVoteResult = []
    for x in fin:
        noOfOnes = 0
        for y in x:
            noOfOnes+=y
        noOfZeros = split-noOfOnes
        if(noOfZeros > noOfOnes):
            majorityVoteResult.append(0)
        else:
            majorityVoteResult.append(1)
#     print( len(majorityVoteResult) , " , " , len(groundTruth))
    acc , p , r , f = evaluate(majorityVoteResult,groundTruth)
    a += acc
    pre += p
    re  += r
    fM += f
    
print(" ACCURACY " , a*100/10)
print(" PRECISION  ", pre*100/10)
print(" RECALL " ,re*100/10)
print(" F MEASURE ", fM*100/10)
# print(e*100/10)
