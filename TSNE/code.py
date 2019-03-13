
# coding: utf-8

# In[ ]:


import time
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

n_sne = 7000

df = pd.read_csv('pca_b.csv')
#Remove the last column
df = df.iloc[:, :-1]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(df)


# In[ ]:


#isolate the last column
dfOrig = pd.read_csv('pca_c.csv')
diseasePD = dfOrig.iloc[:,-1]
disease = np.asarray(diseasePD)


# In[ ]:


#Create a combined file with tSNE vals and disease name
tSNE = pd.DataFrame(tsne_results)

tsne_Combined = tSNE.join(diseasePD)
tsne_Combined.columns = ['tsne1', 'tsne2', 'Disease']
tsne_Combined.to_csv('tsne_Combined.csv')


#Plot from combined file

plotFile = pd.read_csv('tsne_Combined.csv')


#Convert disease name to categorical value (numerical)

plotFile['code'] = pd.factorize(plotFile['Disease'])[0] + 1


# In[ ]:


import seaborn as sns; sns.set()
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

X = plotFile['tsne1']
Y = plotFile['tsne2']
colorby = plotFile['Disease']
mylegend = plotFile.Disease.unique()

plt.figure(figsize=(15,8))
ax = sns.scatterplot(x=X, y=Y, s=40, hue=colorby,data=plotFile, palette = "Set2")


