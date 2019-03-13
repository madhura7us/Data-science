K-means clustering is a NP hard algorithm popular for cluster analysis in data mining. 
K-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster. 
  
Implementation details 

1. Randomly choose data points from the dataset, initialize k and specify maximum number of iterations (or choose by yourself k , centroids and max interations) 
2. Given centroids point and dataset , we iterate over the whole dataset and assign each datapoint to new clusters point on the basis of euclidian distance 
3. The data point with minimum distance belongs to that cluster
4. Then we average all the datapoints belonging to new cluster to find new centroid points 
5. After new centroid points are calculated , we check with current centroid points. If they are equal then we can converge 
6. If it does not converge, update new centroid points with current centroid points and repeat steps 2 , 3 and 4 till it has converged or has reached maximum iterations as per code 
7. K means algorithm iterates over loop and terminates when it is converged or has reached maximum number of iterations given 
 
