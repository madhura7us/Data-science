Hierarchical agglomerative clustering (also called bottom-up clustering) treats each point as a singleton cluster at the outset 
and then successively merge (or agglomerate) pairs of clusters until all clusters have been merged into a single cluster that 
contains all points. 
 
Implementation details 

1. In bottom-up hierarchical clustering, we start with each data item having its own cluster 
2. We then look for the two items that are most similar which is depends of the minimal distance 
3. Then combine them in a larger cluster 
4. We keep repeating until all the clusters we have left are too dissimilar to be gathered together, or until we reach a preset number of clusters 
