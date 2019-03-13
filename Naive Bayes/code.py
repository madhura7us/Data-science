
# coding: utf-8

# In[10]:


import os
import numpy as np
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
from scipy.stats import norm
from collections import defaultdict


# In[19]:


data = pd.read_csv("project3_dataset2.txt", sep='\t',header=None)
vector = data.as_matrix()
vector[:,:-1]


# In[20]:


k = 10
kf = KFold(n_splits=k)
data_types = list(data.dtypes)
accuracy = P = R = F = 0
for trainIndex , testIndex in kf.split(vector):
    
    trainData = vector[trainIndex]
    trainLabel = trainData[:,-1]
#     print(trainData.shape[0])
    trainDict = {}
    
    pTestData = vector[testIndex]
    testLabel = pTestData[:,-1]
    list(np.int_(testLabel))
    testData = pTestData[:,:-1]
#     t = testData.tolist()
#     print(t)
    trainDataDf = pd.DataFrame(trainData)
    groupData = trainDataDf.groupby(len(data.columns)-1)
#     print("here" , groupData[:])
    
    for key,value in groupData:
        trainDict[key] = value.iloc[:,0:len(data.columns)-1]
    # dictonary Comtains label as key and value as whole row of data
    
    meanDict = defaultdict(list)
    
#     print(trainDict)
    
    for key,value in trainDict.items():
        for column in range(len(value.columns)):
#             print(data[column].dtypes)
            if (data_types[column]) == "object":
#                 print("yay")
                meanDict[key].append(0)
            if (data_types[column]) != "object":
#                 print("aayay")
                meanDict[key].append(np.mean(value.iloc[:,column]))
    print("here")
    stdDict = defaultdict(list)
    
    for key,value in trainDict.items():
        for column in range(len(value.columns)):
            if (data_types[column]) == "object":
                stdDict[key].append(0)
            if (data_types[column]) != "object":
                stdDict[key].append(np.std(value.iloc[:,column]))
    
    newLabels = []
    for test in testData:
        maxPDF = 0
        for key,value in trainDict.items():
            pdf = 1
            for col in range(len(test)):
#                 print(data_types[column])
                if data_types[col] != "object":
                    # if it is not of object type we can simly calculate probablity distribution using PDF of the numerical values
#                     print(data_types[column])
                    pdf *= norm.pdf(test[col],meanDict[key][col],stdDict[key][col])
                if data_types[col] == "object":
                    # if dataType is object : it falls in categorical data  
                    tRows, lC, nlC = 0 , 0 , 0
                    # trainDict contains group data in dictonaru format 
                    # key as label of groupData say yes , no or 0 , 1                
                    for keyProb, valueProb in trainDict.items():
                        # calculate totalRows for 0 and 1 datatype
                        # gives count of rows in 0 first , then in 1 ;
                        tRows += valueProb.count()[col]
                        for val in valueProb.iloc[:,col]:
                            # iterate over all values in dictionary 
                            # check if its value matches with the given category
                            if val == test[col]:
                                # then if it matches with category the column belongs to
                                # if true we increment labelCOunt as 1
                                if keyProb == key:
                                    lC += 1
                                #else if its key doesnot match , their final result match but it falls in different category
                                else:
                                    nlC += 1
                    # we calculate the probabilty        
                    probability = (float(lC) / (value.iloc[:,col].count())) / (float(lC + nlC) / float(tRows))  
                    # append it to pdf 
                    pdf *= float(value.count()[0] / trainData.shape[0]) * probability  
#                     pdf *= float(value.count()[0] / trainData.shape[0]) * probability 
#           condition for maximum probabilty
            if pdf>maxPDF:
#                 if new PDf is greater than before update  new pdf and update  the marked output value 
                maxPDF = pdf
#                  new value updated
                newGuess = key
        newLabels.append(newGuess)
#     print(newLabels)
#     print(len(testLabel))
#     print(len(newLabels))
#     testLabel = list(testLabel)
    tp=0 
    tn=0
    fp=0
    fn=0
    
#     print(newLabels)
#     print(testLabel)
    
    for predicted,actual in zip(newLabels,testLabel):
        if predicted == 1 and actual == 1.0:
            tp += 1
        elif predicted == 0 and actual == 0.0:
            tn += 1
        elif predicted == 0 and actual == 1.0:
            fn += 1
        elif predicted == 1 and actual == 0.0:
            fp += 1 
    accuracy += (float)(tp+tn)/(float)(tp+fp+tn+fn)
    P += (float)(tp)/(float)(tp+fp)
    R += (float)(tp)/(float)(tp+fn)
    F = ((float)(2*P*R)/(float)(R+P))
    print(tp,fp,tn,fn)
print("Accuracy: ", accuracy * 100 / k )
print("Precision: ", P * 100 / k )
print("Recall: ", R * 100 / k )
print("F-Measure: ", F * 100 / k ) 


# In[21]:


lstAA = [[132, 6.2, 6.47, 36.21, 'Present', 62, 30.77, 14.14, 45], [123, 0.05, 4.61, 13.69, 'Absent', 51, 23.23, 2.78, 16], [128, 0.5, 3.7, 12.81, 'Present', 66, 21.25, 22.73, 28], [114, 9.6, 2.51, 29.18, 'Absent', 49, 25.67, 40.63, 46], [150, 0.3, 6.38, 33.99, 'Present', 62, 24.64, 0.0, 50], [136, 8.8, 4.69, 36.07, 'Present', 38, 26.56, 2.78, 63], [144, 0.76, 10.53, 35.66, 'Absent', 63, 34.35, 0.0, 55], [134, 11.79, 4.01, 26.57, 'Present', 38, 21.79, 38.88, 61], [126, 8.75, 6.53, 34.02, 'Absent', 49, 30.25, 0.0, 41], [164, 5.6, 3.17, 30.98, 'Present', 44, 25.99, 43.2, 53], [178, 20.0, 9.78, 33.55, 'Absent', 37, 27.29, 2.88, 62], [136, 3.99, 2.58, 16.38, 'Present', 53, 22.41, 27.67, 36], [128, 0.0, 2.63, 23.88, 'Absent', 45, 21.59, 6.54, 57], [146, 0.64, 4.82, 28.02, 'Absent', 60, 28.11, 8.23, 39], [112, 4.46, 7.18, 26.25, 'Present', 69, 27.29, 0.0, 32], [174, 2.02, 6.57, 31.9, 'Present', 50, 28.75, 11.83, 64], [162, 0.0, 5.09, 24.6, 'Present', 64, 26.71, 3.81, 18], [216, 0.92, 2.66, 19.85, 'Present', 49, 20.58, 0.51, 63], [142, 1.32, 7.63, 29.98, 'Present', 57, 31.16, 72.93, 33], [160, 12.0, 5.73, 23.11, 'Present', 49, 25.3, 97.2, 52], [134, 0.0, 5.63, 29.12, 'Absent', 68, 32.33, 2.02, 34], [138, 2.0, 5.11, 31.4, 'Present', 49, 27.25, 2.06, 64], [178, 0.95, 4.75, 21.06, 'Absent', 49, 23.74, 24.69, 61], [118, 0.12, 1.96, 20.31, 'Absent', 37, 20.01, 2.42, 18], [128, 5.16, 4.9, 31.35, 'Present', 57, 26.42, 0.0, 64], [176, 5.76, 4.89, 26.1, 'Present', 46, 27.3, 19.44, 57], [170, 4.2, 4.67, 35.45, 'Present', 50, 27.14, 7.92, 60], [146, 6.4, 5.62, 33.05, 'Present', 57, 31.03, 0.74, 46], [132, 0.0, 1.87, 17.21, 'Absent', 49, 23.63, 0.97, 15], [150, 3.5, 6.99, 25.39, 'Present', 50, 23.35, 23.48, 61], [156, 0.0, 3.47, 21.1, 'Absent', 73, 28.4, 0.0, 36], [138, 8.8, 3.12, 22.41, 'Present', 63, 23.33, 120.03, 55], [168, 9.0, 8.53, 24.48, 'Present', 69, 26.18, 4.63, 54], [168, 4.5, 6.68, 28.47, 'Absent', 43, 24.25, 24.38, 56], [112, 0.0, 1.71, 15.96, 'Absent', 42, 22.03, 3.5, 16], [110, 2.35, 3.36, 26.72, 'Present', 54, 26.08, 109.8, 58], [132, 6.0, 5.97, 25.73, 'Present', 66, 24.18, 145.29, 41], [136, 7.36, 2.19, 28.11, 'Present', 61, 25.0, 61.71, 54], [128, 1.6, 5.41, 29.3, 'Absent', 68, 29.38, 23.97, 32], [108, 15.0, 4.91, 34.65, 'Absent', 41, 27.96, 14.4, 56], [134, 3.0, 3.17, 17.91, 'Absent', 35, 26.37, 15.12, 27], [130, 1.75, 5.46, 34.34, 'Absent', 53, 29.42, 0.0, 58], [128, 2.6, 4.94, 21.36, 'Absent', 61, 21.3, 0.0, 31], [130, 2.78, 4.89, 9.39, 'Present', 63, 19.3, 17.47, 25], [126, 0.96, 4.99, 29.74, 'Absent', 66, 33.35, 58.32, 38], [114, 3.6, 4.16, 22.58, 'Absent', 60, 24.49, 65.31, 31], [136, 6.6, 6.08, 32.74, 'Absent', 64, 33.28, 2.72, 49]]


# In[22]:


lstBB =[[132, 6.2, 6.47, 36.21, 'Present', 62, 30.77, 14.14, 45], [123, 0.05, 4.61, 13.69, 'Absent', 51, 23.23, 2.78, 16], [128, 0.5, 3.7, 12.81, 'Present', 66, 21.25, 22.73, 28], [114, 9.6, 2.51, 29.18, 'Absent', 49, 25.67, 40.63, 46], [150, 0.3, 6.38, 33.99, 'Present', 62, 24.64, 0.0, 50], [136, 8.8, 4.69, 36.07, 'Present', 38, 26.56, 2.78, 63], [144, 0.76, 10.53, 35.66, 'Absent', 63, 34.35, 0.0, 55], [134, 11.79, 4.01, 26.57, 'Present', 38, 21.79, 38.88, 61], [126, 8.75, 6.53, 34.02, 'Absent', 49, 30.25, 0.0, 41], [164, 5.6, 3.17, 30.98, 'Present', 44, 25.99, 43.2, 53], [178, 20.0, 9.78, 33.55, 'Absent', 37, 27.29, 2.88, 62], [136, 3.99, 2.58, 16.38, 'Present', 53, 22.41, 27.67, 36], [128, 0.0, 2.63, 23.88, 'Absent', 45, 21.59, 6.54, 57], [146, 0.64, 4.82, 28.02, 'Absent', 60, 28.11, 8.23, 39], [112, 4.46, 7.18, 26.25, 'Present', 69, 27.29, 0.0, 32], [174, 2.02, 6.57, 31.9, 'Present', 50, 28.75, 11.83, 64], [162, 0.0, 5.09, 24.6, 'Present', 64, 26.71, 3.81, 18], [216, 0.92, 2.66, 19.85, 'Present', 49, 20.58, 0.51, 63], [142, 1.32, 7.63, 29.98, 'Present', 57, 31.16, 72.93, 33], [160, 12.0, 5.73, 23.11, 'Present', 49, 25.3, 97.2, 52], [134, 0.0, 5.63, 29.12, 'Absent', 68, 32.33, 2.02, 34], [138, 2.0, 5.11, 31.4, 'Present', 49, 27.25, 2.06, 64], [178, 0.95, 4.75, 21.06, 'Absent', 49, 23.74, 24.69, 61], [118, 0.12, 1.96, 20.31, 'Absent', 37, 20.01, 2.42, 18], [128, 5.16, 4.9, 31.35, 'Present', 57, 26.42, 0.0, 64], [176, 5.76, 4.89, 26.1, 'Present', 46, 27.3, 19.44, 57], [170, 4.2, 4.67, 35.45, 'Present', 50, 27.14, 7.92, 60], [146, 6.4, 5.62, 33.05, 'Present', 57, 31.03, 0.74, 46], [132, 0.0, 1.87, 17.21, 'Absent', 49, 23.63, 0.97, 15], [150, 3.5, 6.99, 25.39, 'Present', 50, 23.35, 23.48, 61], [156, 0.0, 3.47, 21.1, 'Absent', 73, 28.4, 0.0, 36], [138, 8.8, 3.12, 22.41, 'Present', 63, 23.33, 120.03, 55], [168, 9.0, 8.53, 24.48, 'Present', 69, 26.18, 4.63, 54], [168, 4.5, 6.68, 28.47, 'Absent', 43, 24.25, 24.38, 56], [112, 0.0, 1.71, 15.96, 'Absent', 42, 22.03, 3.5, 16], [110, 2.35, 3.36, 26.72, 'Present', 54, 26.08, 109.8, 58], [132, 6.0, 5.97, 25.73, 'Present', 66, 24.18, 145.29, 41], [136, 7.36, 2.19, 28.11, 'Present', 61, 25.0, 61.71, 54], [128, 1.6, 5.41, 29.3, 'Absent', 68, 29.38, 23.97, 32], [108, 15.0, 4.91, 34.65, 'Absent', 41, 27.96, 14.4, 56], [134, 3.0, 3.17, 17.91, 'Absent', 35, 26.37, 15.12, 27], [130, 1.75, 5.46, 34.34, 'Absent', 53, 29.42, 0.0, 58], [128, 2.6, 4.94, 21.36, 'Absent', 61, 21.3, 0.0, 31], [130, 2.78, 4.89, 9.39, 'Present', 63, 19.3, 17.47, 25], [126, 0.96, 4.99, 29.74, 'Absent', 66, 33.35, 58.32, 38], [114, 3.6, 4.16, 22.58, 'Absent', 60, 24.49, 65.31, 31], [136, 6.6, 6.08, 32.74, 'Absent', 64, 33.28, 2.72, 49]]


# In[23]:


count = 0
for x,y in zip(lstAA,lstBB):
    if(x==y):
        count+=1
#         print("XX")
    else:
        count+=1
        print(count)


# In[24]:


count


# In[25]:


len(lstAA)


# In[26]:





# In[ ]:


len()


# In[ ]:


k = 10
kf = KFold(n_splits=k)
data_types = list(data.dtypes)
accuracy = P = R = F = 0
for trainIndex , testIndex in kf.split(vector):
    
    trainData = vector[trainIndex]
    trainLabel = trainData[:,-1]
    
    trainDict = defaultdict()
    
    pTestData = vector[testIndex]
    testLabel = pTestData[:,-1]
    list(np.int_(testLabel))
    testData = pTestData[:,:-1]
    
    groupData = data.groupby(len(data.columns)-1)
#     print("here")
    
    for key,value in groupData:
        trainDict[key] = value.iloc[:,0:len(data.columns)-1]
    
    meanDict = defaultdict(list)
    
    
    for key,value in trainDict.items():
        for column in range(len(value.columns)):
#             print(data[column].dtypes)
            if (data_types[column]) == "object":
#                 print("yay")
                meanDict[key].append(0)
            if (data_types[column]) != "object":
#                 print("aayay")
                meanDict[key].append(np.mean(value.iloc[:,column]))
    print("here")
    stdDict = defaultdict(list)
    
    for key,value in trainDict.items():
        for column in range(len(value.columns)):
            if (data_types[column]) == "object":
                stdDict[key].append(0)
            if (data_types[column]) != "object":
                stdDict[key].append(np.std(value.iloc[:,column]))
    
    newLabels = []
    for test in testData:
        maxPDF = 0
        for key,value in trainDict.items():
            pdf = 1
            for col in range(len(test)):
#                 print(data_types[column])
                if (data_types[col]) != "object":
#                     print(data_types[column])
                    pdf *= norm.pdf(test[col],meanDict[key][col],stdDict[key][col])
                if (data_types[col]) == "object":
# #                     pdf = 1
#                     total_rows = 0
#                     count_value_label = 0
#                     count_value_non_label = 0
#                     for label1, data1 in trainDict.items():
#                         total_rows += data1.count()[col]
#                         for val in data1.iloc[:,col]:
#                             if val == test[col] and label1 == key:
#                                 count_value_label += 1
#                             elif val == test[col]:
#                                 count_value_non_label += 1

#                     numerator = (float(count_value_label) / value.iloc[:,col].count())
#                     denominator = float(count_value_label + count_value_non_label) / float(total_rows) 
#                     probability = numerator / denominator
#                     pdf *= float(value.count()[0] / total_rows) * probability    
            if pdf>maxPDF:
                maxPDF = pdf
                newGuess = key
        newLabels.append(newGuess)
#     print(len(testLabel))
#     print(len(newLabels))
#     testLabel = list(testLabel)
    tp=0 
    tn=0
    fp=0
    fn=0
    
#     print(newLabels)
#     print(testLabel)
    
    for predicted,actual in zip(newLabels,testLabel):
        if predicted == 1 and actual == 1.0:
            tp += 1
        elif predicted == 0 and actual == 0.0:
            tn += 1
        elif predicted == 0 and actual == 1.0:
            fn += 1
        elif predicted == 1 and actual == 0.0:
            fp += 1 
    accuracy += (float)(tp+tn)/(float)(tp+fp+tn+fn)
    P += (float)(tp)/(float)(tp+fp)
    R += (float)(tp)/(float)(tp+fn)
    F = 2*((float)(P*R)/(float)(R+P))
    print(tp,fp,tn,fn)
print("Accuracy: ", accuracy * 100 / k )
print("Precision: ", P * 100 / k )
print("Recall: ", R * 100 / k )
print("F-Measure: ", F * 100 / k ) 

