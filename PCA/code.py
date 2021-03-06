
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#convert txt to csv and read

txt_file='pca_c.txt'
csv_table=pd.read_table(txt_file,sep='\t')
csv_table.to_csv('pca_c.csv',index=False, header=None)

#read all but the last column

df = pd.read_csv('pca_c.csv', header=None)
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) #remove whitespaces

df = df.iloc[:, :-1]
#df


# In[ ]:


#calculate the mean of each column

cols =len(df.columns)

meanArr = []
for i in range(0,cols):
        meanArr.append(df[i].mean())


# In[ ]:


#subtract the mean of each column from all vals in that column

rows = df[0].count()
for i in range(0,cols):
    for j in range(0,rows):
        df.iloc[j][i] = df.iloc[j][i] - meanArr[i]


# In[ ]:


#calculate the covariance matrix of the dataframe

covMat = df.cov()


# In[ ]:


#calculate eigenvectors and eigenvalues

from numpy import linalg as LA

eigenval, eigenvec = LA.eig(covMat)


# In[ ]:


#sort eigenvectors by eigenvals in descending order

ev_list = zip(eigenval, eigenvec)
ev_list = sorted(ev_list, key=lambda tup:tup[0], reverse=True)
eigenval, eigenvec = zip(*ev_list)


# In[ ]:


#select the 2 highest dimensions

dim1 = eigenvec[0]
dim2 = eigenvec[1]


# In[ ]:


#append into a matrix and find its transpose
PCA = []
PCA.append(eigenvec[0])
PCA.append(eigenvec[1])

PCArr = np.squeeze(np.asarray(PCA)) #convert matrix to numpy array

PCATr = PCArr.transpose()
PCATr

#matrix multiplication
PC = np.dot(df, PCATr)
np.savetxt("PCVals_c.csv", PC, delimiter=",") #write PC to file
#PC


# In[ ]:


#Re-read original csv for disease names

dfOrig = pd.read_csv('pca_c.csv', header=None)


# In[ ]:


#isolate the last column
diseasePD = dfOrig.iloc[:,-1]
disease = np.asarray(diseasePD)
disease.shape
PC.shape


# In[ ]:


#Create a combined file with PC vals and disease name
PCPD = pd.DataFrame(PC)

PC_Combined = PCPD.join(diseasePD)
PC_Combined.columns = ['PC1', 'PC2', 'Disease']
PC_Combined.to_csv('PC_Combined.csv')


# In[ ]:


#Plot from combined file

plotFile = pd.read_csv('PC_Combined.csv')


#Convert disease name to categorical value (numerical)

plotFile['code'] = pd.factorize(plotFile['Disease'])[0] + 1


# In[ ]:


import seaborn as sns; sns.set()
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

X = plotFile['PC1']
Y = plotFile['PC2']
colorby = plotFile['Disease']
mylegend = plotFile.Disease.unique()

plt.figure(figsize=(15,8))
ax = sns.scatterplot(x=X, y=Y, s=40, hue=colorby,data=plotFile, palette = "Set2")


