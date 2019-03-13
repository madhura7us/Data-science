Density-based spatial clustering of applications with noise (DBSCAN) is a clustering algorithm based on distance to neighboring points, a radial distance (epsilon), and a certain threshold of points required to form a cluster(minPts). 
Selection of epsilon and minPts is highly based on the dataset. Often epsilon selection is done based on the curve of the KNN plot of the data. 
In this case however, a trial error approach has been used to select the values for the parameters. 

Implementation details 

1. Set epsilon and minPts value 
2. Initialize clusterCount to 0 
3. Read the dataset and convert to a matrix format with only the point vectors 
4. Calculate the distance matrix In the dbscan function, call the initial_label_points function, mark all points as undefined to begin with. Then choose point P and call the rangeQuery function. 
5. In the rangeQuery function, calculate the distance from point P to all other points and return neighbors within epsilon distance from P 
6. If the number of neighbors < minPts, mark P as noise. Otherwise mark P as core and add it to the cluster 
7. If P is core, call the rangeQuery function for every neighbor of P and mark noise points as border points and add to the original cluster. If a neighbor is found to be a core point, increment clusterCount and add neighbor to that cluster
8. Repeat from step 5 till all points in the dataset have been marked 
