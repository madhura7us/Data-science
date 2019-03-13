Algorithm 
-Choose T—number of trees to grow 
-Choose m<M (M is the number of total features)
-number of features used to calculate the best split at each node
-For each tree
  • Choose a training set by choosing N times (N is the number of training examples) with replacement from the training set
  • For each node, randomly choose m features and calculate the best split 
  • Fully grown and not pruned 
 -Use majority voting among all the trees 
 
Implementation details 
1. Read the data in csv format. 
2. Build decision tree. 
3. First, we use k fold validation to break the data randomly in train and test data. 
4. After k split, we get trained and test indexes, using that we build our train data and test data list. 
5. Then train data is split into further small dataset to generate separate trees, which will be our forest(bagging). Each time while generating new data split, we also choose m attributes which are generally 20% of the given total attributes and must be less than total number of attributes. 
6. Similarly, we make list of test set with similar attributes, and also we make list of ground truth values to check accuracy, precision, recall and f measure. 
7. After we have train data split, we build and store our trees in a list (train data set with m attributes chosen randomly). 
8. After our trees are generated, we iterate over our test result with specific decision tree built with those m attributes and make result list with all the results obtained from decision tree over test result.
9. Finally, we get result set, we iterate over it and by using majority voting assign class label to the test data.  
10. After this we have new label list and ground truth, we calculate true positive, true negative , false positive, false negative, and using that we measure accuracy, precision and recall and f- measure. 

Parameter selection 
1. We use K FOLD to split train data and test data 
2. We split our train data into further small data set, here number of small dataset is preferred to be odd, because while testing the result will be from odd number of list for single test data, which will not bring any tie condition during majority voting 
3. Number of attributes chosen randomly, are less than total number of attributes. 
 
Handling continuous and categorical features 
A major advantage of decision trees is that categorical values are treated as any other value. 
The only change we made for processing continuous and categorical values was in the comparison condition (which is set to ‘>=’ for continuous and ‘==’ for categorical values respectively). 
