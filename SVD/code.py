
# coding: utf-8

# In[1]:


from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd

txt_file='pca_a.txt'
csv_table=pd.read_table(txt_file,sep='\t')
csv_table.to_csv('pca_a.csv',index=False, header=None)


# In[2]:



myData = pd.read_csv('pca_a.csv', header=None)
myData = myData.iloc[:, :-1]
u, s, vt = svds((myData - myData.mean()).iloc[:,:-1],2)

pc = u @ np.diag(s)
pc = pc[:,::-1]

explained_variance = np.var(pc, axis=0)
full_variance = np.var((myData - myData.mean()).iloc[:,:-1], axis=0)
explained_variance_ratio = explained_variance / full_variance.sum()
ax = pd.Series(explained_variance_ratio.cumsum()).plot(
                    kind='line', figsize=(10,4)).set_ylim([0,1])

explained_variance_ratio.cumsum()



# In[3]:


import seaborn as sns
X = pd.read_csv('pca_a.csv', header = None)
disease = X.iloc[:, -1]
myData = myData.join(disease, how='left', lsuffix='_left', rsuffix='_right')

myData = myData.rename(columns={ myData.columns[-1]: "Disease" })

myData_svd = pd.concat((pd.DataFrame(pc, index=myData.index
                        , columns=('c0','c1')), myData.loc[:,'Disease']),1)

g = sns.lmplot('c0', 'c1', myData_svd, hue='Disease', fit_reg=False, size=8
              ,scatter_kws={'alpha':0.7,'s':60})

