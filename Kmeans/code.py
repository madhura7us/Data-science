
# coding: utf-8

# # K MEANS ALGORITHM 

# Adding all the libraries

# In[1]:


import sys
import pandas as pd
import numpy as np
import random
# from scipy.spatial import distance 
from collections import defaultdict
import seaborn as sns; sns.set()  # for plot styling
import operator


# READIND DATA FROM THE IYER FILE and Convertin it into numpy matrix 

# In[2]:


data = pd.read_csv("newDataSet.txt" , sep= '\t' , header=None)
# print(data.head(10))  # print first 10 result of data  

#
gene_vector = data.as_matrix()
gene_vector.shape
# len(gene_vector)
# gene_vector.shape
# data
gene_vector[0][2:gene_vector.shape[1]]


# __CREATING VECTORS and inserting into list after removing outliers__

# In[3]:


groundtruth = []
groundDict = defaultdict()
for i in range(0,len(gene_vector)):
    groundtruth.append(gene_vector[i][1])


# In[4]:


gene_vector.shape


# In[5]:


lst = []
for i in range(0,len(gene_vector)):
    lst.append(gene_vector[i][2:gene_vector.shape[1]])
len(lst)
len(lst[0])


# __CHOOSING RANDOMLY K POINTS FOR K MEAN USING (write method name from wiki page )__

# In[6]:


k = 5
centroids = []
lstPoint = [14,1,50,300,89]
noItr = 30
for l in lstPoint:
    centroids.append(gene_vector[l][2:gene_vector.shape[1]])
centroids
# for i in range(0,k):
#     ra =  random.randrange(0,gene_vector.shape[0])
#     print(ra)
#     centroids.append(gene_vector[ra][2:gene_vector.shape[1]])
# centroids


# In[7]:


# DEFING DICTONARY TO STORE POINTS
d = defaultdict(list)
for c in range(0,len(centroids)):
    d[c] = []
len(d)


# In[8]:


# while(True):
#     for x in lst:
#         for c in centroids:
#             lstDist.append(round(np.linalg.norm(x-c)))
#         minIndex = 50000
#         minIndex = lstDist.index(min(lstDist))   #
#         d[minIndex].append(x)  # append point to that centroid point
#         newCentroidLst.append(minIndex)
#         lstDist.clear()


# STEP 2 :  Assign each x_ix i to nearest cluster by calculating its distance to each centroid.

# In[9]:


# WORKING ON STEP TWO 
lstDist = []
i = 0
avgLst = []
newCentroidLst = []

while(True):
    print(" iteration "  , i)
  
#     print(centroids)
    for x in lst:
        for c in centroids:
            lstDist.append((np.linalg.norm(x-c)))
        minIndex = lstDist.index(min(lstDist))   #
#         print(" LIST " , lstDist)
#         print(minIndex)
        d[minIndex].append(x)  # append point to that centroid point
        newCentroidLst.append(minIndex+1)
        lstDist.clear()
        
    for c in d:
        avg = np.average(d[c] , axis=0)
        avgLst.append(avg)
    count = 0 
    for x in range(0,len(centroids)):
        if(np.array_equal(centroids[x],avgLst[x]) == True):
            count+=1
    print(count)
    if(count == len(d) or i>=maxItr):
        break;
    else:
        newCentroidLst.clear()
    centroids = list(avgLst) #newCentroids
    avgLst.clear()
    d.clear()
    i+=1


# In[10]:


# # WORKING ON STEP TWO 
# lstDist = []
# i = 0
# avgLst = []
# newCentroidLst = []
# while(True):
#     print(" asd "  , i)
#     i+=1
# #     print(centroids)
#     for x in lst:
#         for c in centroids:
#             lstDist.append(round(np.linalg.norm(x-c)))
#         minIndex = lstDist.index(min(lstDist))   #
#         d[minIndex].append(x)  # append point to that centroid point
#         newCentroidLst.append(minIndex)
#         lstDist.clear()
        
#     for c in d:
#         avg = np.average(d[c] , axis=0)
#         avgLst.append(avg)
        
#     truTable = []
#     for x in range(0,len(centroids)):
#         truTable.append( list(np.equal(centroids[x],avgLst[x])))

#     tTable = [True]*len(centroids[0])
#     count = 0 
#     for t in truTable:
#         if(tTable == t):
#             count+=1 
    
#     if(count == 5 or i > noItr):
#         break
#     else:
#         newCentroidLst.clear()
#     truTable.clear()
#     centroids = list(avgLst) #newCentroids
#     avgLst.clear()
#     d.clear()
# #     


# In[11]:


for k in d:
    print(len(d[k]))


# In[12]:


groundMat = np.zeros( (len(groundtruth),len(groundtruth)) ,  dtype=int )
for i in range(0,len(groundtruth)):
    for j in range(0,len(groundtruth)):
        if(groundtruth[i] == groundtruth[j] ):
            groundMat[i][j] = 1
        else:
            groundMat[i][j] = 0
#             print("y")


# In[13]:


clusterMat = np.zeros( (len(newCentroidLst),len(newCentroidLst)) ,  dtype=int )
for i in range(0,len(newCentroidLst)):
    for j in range(0,len(newCentroidLst)):
        if(newCentroidLst[i] == newCentroidLst[j] ):
            clusterMat[i][j] = 1
        else:
            clusterMat[i][j] = 0


# In[14]:


len(lst)


# In[15]:


# groundDict['[1.   0.72 0.1  0.57 1.08 0.66 0.39 0.49 0.28 0.5  0.66 0.52]']


# In[16]:


# # MAKING GROUND MATRIX 

# groundMat = np.zeros(shape=(len(groundDict),len(groundDict)))

# for i in range(0,len(groundDict)):
# #     print("Y")
#     valA = groundDict[str(lst[i])]         
#     for j in range(0,len(groundDict)):
# #         print(i,j)
#         valB = groundDict[str(lst[j])]
#         if( valA == valB ):
#             groundMat[i][j] = 1
#         else:
#             groundMat[i][j] = 0

# # print("X")

# # MAKING CLUSTER MATRIX

# clusterMat = np.zeros(shape=(len(clusterDict),len(clusterDict)))

# for i in range(0,len(clusterDict)):
#     valA = clusterDict[str(lst[i])]         
#     for j in range(0,len(clusterDict)):
#         valB = clusterDict[str(lst[j])]
#         if( valA == valB ):
#             clusterMat[i][j] = 1
#         else:
#             clusterMat[i][j] = 0



# In[ ]:





# In[17]:


groundMat


# In[18]:


clusterMat


# In[19]:


M00 = 0
M01 = 0
M10 = 0
M11 = 0
for i in range(0,len(clusterMat)):
    for j in range(0,len(clusterMat)):
        if( groundMat[i][j]==0 and clusterMat[i][j] == 0 ):
            M00+=1
        if(  groundMat[i][j]==0 and clusterMat[i][j] == 1 ):
            M01+=1
        if( groundMat[i][j]==1 and clusterMat[i][j] == 0 ):
            M10+=1
        if(  groundMat[i][j]==1 and clusterMat[i][j] == 1 ):
            M11+=1
            
    
print("was" , " " , M00)

print("was" , " " , M01)


print("was" , " " , M10)


print("was" , " " , M11)
            


# In[20]:


from sklearn.decomposition import PCA  


# In[21]:


pca = PCA(n_components=2)
df_transform = pca.fit_transform(lst)


# In[22]:


import os
import matplotlib.pyplot as plt
clabels = np.asarray(newCentroidLst).reshape(len(newCentroidLst),1)
print(len(clabels) , len(df_transform))
mat_labels = np.append(df_transform , clabels, axis=1)
# mat_labels

final_df = pd.DataFrame(mat_labels)
final_df.columns = ['x','y','label']
final_df.label = final_df.label.astype(int)

def plotScatter(final_df,filename):
    groups = final_df.groupby('label')
    fig, ax = plt.subplots()
    ax.margins(0.05)
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name)
    ax.set_title(filename)   
    ax.legend()
    
    fig_size = [12,9]
    plt.rcParams["figure.figsize"] = fig_size
    
    plt.xlabel("principal component 1")
    plt.ylabel("principal component 2")

    plt.show()
    
plotScatter(final_df,"PCA plot for " + os.path.basename('newDataSet.txt') )


# In[23]:


print("    jaccard      " , (M11)/(M11+M01+M10) )
print( "   random       " , (M11+M00)/(M11+M01+M10+M00))

