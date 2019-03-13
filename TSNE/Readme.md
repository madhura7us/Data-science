Dataset: 

Provided for implementation (pca_a.txt, pca_b.txt, pca_c.txt)

Pre-processing: 

1. Converted text files to csv  
2. Removed whitespaces while reading to dataframe 
3. Dropped last column (disease names) since they are not need for PCA computation

Implementation: 
Here is the step-by-step implementation of the PCA algorithm:
 
1. Calculated the mean of each column (feature) 
2. Centralized the data by subtracting the mean from each value 
3. Calculated the covariance matrix of the dataframe 
4. Computed the eigenvalues and eigenvectors 
5. Sorted eigenvectors by eigenvalues in descending order 
6. Selected the top two dimensions 
7. Appended the two components into a matrix and computed its transpose 
8. Projected the components onto original data 
9. Read disease column into another dataframe  
10. Converted disease names into categorical values 
11. Appended the PC values and categorical disease values 
12. Plotted a scatterplot using seaborn colored as per disease 
 
 
