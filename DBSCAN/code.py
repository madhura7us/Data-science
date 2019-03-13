
# coding: utf-8




import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform
##############################################################

eps = 1.03
minPts = 5
clusterCount = 0

txt_file='cho.txt'
csv_table=pd.read_table(txt_file,sep='\t', header = None)
csv_table.to_csv('cho.csv', index=False)

# Get rid of unnecessary columns
df = pd.read_csv('cho.csv', header=None)
#print(len(df))
df = df.drop([0,1], axis = 1)
#print(len(df))
df = df.drop(df.index[0])
# print(len(df))

# Convert df to matrix

DataMat = df.as_matrix()
#print(DataMat)

# Calculate distance matrix where each row is a point


tempMat = pdist(DataMat, metric='euclidean')
distMat = squareform(tempMat, force='no', checks=True)
#print(distMat)

##############################################################





# Algorithm
#DBSCAN(DB, distFunc, eps, minPts) {
#    C = 0                                                  /* Cluster counter */
#    for each point P in database DB {
#       if label(P) ≠ undefined then continue               /* Previously processed in inner loop */
#       Neighbors N = RangeQuery(DB, distFunc, P, eps)      /* Find neighbors */
#       if |N| < minPts then {                              /* Density check */
#          label(P) = Noise                                 /* Label as Noise */
#          continue
#       }
#       C = C + 1                                           /* next cluster label */
#       label(P) = C                                        /* Label initial point */
#       Seed set S = N \ {P}                                /* Neighbors to expand */
#       for each point Q in S {                             /* Process every seed point */
#          if label(Q) = Noise then label(Q) = C            /* Change Noise to border point */
#          if label(Q) ≠ undefined then continue            /* Previously processed */
#          label(Q) = C                                     /* Label neighbor */
#          Neighbors N = RangeQuery(DB, distFunc, Q, eps)   /* Find neighbors */
#          if |N| ≥ minPts then {                           /* Density check */
#             S = S ∪ N                                     /* Add new neighbors to seed set */
#          }
#       }
#    }
# }

def dbscan(distMat, eps, minPts, clusterCount):
    
    # Set all points as undefined initially
    initial_label_points(distMat)
    
    s = set()
    newNeighList = []
    for i  in range(0, len(distMat)):
        point = i
        
        if(point_labels[point] != 'undefined'):
            continue
            
            
        # Find neighbors of point    
        Neighbors = rangeQuery(distMat, eps, minPts, point)
        
        
        N = len(Neighbors)
        print("Neighbors len", N)
        
        
        if (N < minPts):
            #print("In N < minPts")
            point_labels[point] = 'noise'
            #print("Line 36 {}".format(point_labels))
            continue

        else:
            clusterCount += 1
            
            point_labels[point] = str(clusterCount)
            
            
            for q in Neighbors:
                if(point_labels[q] == 'noise'):
                        point_labels[q] = str(clusterCount)
                elif(point_labels[q] == 'undefined'):
                        point_labels[q] = str(clusterCount)

               
                        qNeighbors = rangeQuery(distMat, eps, minPts, q)

                        qN = len(qNeighbors)
                        if qN >= minPts:
                    
                            Neighbors += qNeighbors
    print(point_labels)
    return clusterCount, point_labels







# RangeQuery(DB, distFunc, Q, eps) {
#    Neighbors = empty list
#    for each point P in database DB {                      /* Scan all points in the database */
#       if distFunc(Q, P) ≤ eps then {                      /* Compute distance and check epsilon */
#          Neighbors = Neighbors ∪ {P}                      /* Add to result */
#       }
#    }
#    return Neighbors
# }

def rangeQuery(distMat, eps, minPts, point):
    neighbors = []
    # Find distance from point to all other points
    print(point)
    dist_to_all = distMat[:, point]
    
    for i in range(0, len(dist_to_all)):
        
        if (dist_to_all[i] <= eps):
            neighbors.append(i)
    #print("rangeQuery Neighbors: ", neighbors)
    return neighbors
            





def initial_label_points(distMat):
    global point_labels
    point_labels = {}
    for i  in range(0, len(distMat)):
        point_labels[i] = 'undefined'





count, labels = dbscan(distMat, eps, minPts, clusterCount)





# print(labels)




print('Points ClusterID')
for points, clusterID in point_labels.items():
    print(' {}    {}'.format(points, clusterID))




v = {}

for points, clusterID in sorted(point_labels.items()):
    v.setdefault(clusterID, []).append(points)





# print(v)





def calc_jaccard_rand(ground_truth, cluster_labels):
    m00 = 0
    m01 = 0
    m10 = 0
    m11 = 0

    for i in range(len(ground_truth)):
        for j in range(len(cluster_labels)):
            if (ground_truth[i] != ground_truth[j]) and (cluster_labels[i] != cluster_labels[j]):
                m00 += 1
            elif (ground_truth[i] != ground_truth[j]) and (cluster_labels[i] == cluster_labels[j]):
                m01 += 1
            elif (ground_truth[i] == ground_truth[j]) and (cluster_labels[i] != cluster_labels[j]):
                m10 += 1
            elif (ground_truth[i] == ground_truth[j]) and (cluster_labels[i] == cluster_labels[j]):
                m11 += 1

    jacc = float(m11) / float(m11 + m10 + m01)
    rand = float(m11 + m00) / float(m11 + m10 + m01 + m00)
    print("Jaccard: ",jacc)
    print("Rand: ", rand)





df2 = pd.read_csv(txt_file, sep='\t',header=None)
pred = []

ground_truth = list(np.reshape(df2.iloc[:,1].values,[df2.shape[0],1]).flat)
for i in range(len(labels)):
    if labels[i] == 'noise':
        labels[i] = -1
        
    else:
        labels[i] = int(labels[i])
    
    pred.append(labels[i])

# print(len(ground_truth))
# print(len(pred))

calc_jaccard_rand(ground_truth, pred)



from sklearn.decomposition import PCA

df = pd.read_csv(txt_file, sep='\t', header=None)
dfMat = np.asmatrix(df.iloc[:,2:18])

pca = PCA(n_components=2)
df_transform = pca.fit_transform(dfMat)
df_transform[:6]

clabels = np.asarray(pred).reshape(len(pred),1)
MatLabels = np.append(df_transform, clabels, axis=1)
#print(MatLabels)

final_df = pd.DataFrame(MatLabels)
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
    
plotScatter(final_df,"PCA plot for " + os.path.basename(txt_file))



