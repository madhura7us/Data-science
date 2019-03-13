DECISION TREE 
Algorithm 
1. Start at the root node.  
2. For each X, find the set S that minimizes the sum of the node impurities in the two child nodes and choose the split {X∗ ∈ S∗} that gives the minimum overall X and S.  
3. If a stopping criterion is reached, exit. Otherwise, apply step 2 to each child node in turn. 
 
Implementation details 
1. Load data and split using K-folds (with k = 10) 
2. Calculate the gini to measure the impurity in the dataset 
3. Based on a condition, split the dataset into true and false rows 
4. Calculate the information gain of the node 
5. Pick the condition with the most information gain 
6. Recursively build the tree 
7. Measure the performance 
 
Parameter selection 

The only parameter is the number of folds to use 

Handling continuous and categorical features 
A major advantage of decision trees is that categorical values are treated as any other value. 
The only change we made for processing continuous and categorical values was in the comparison condition (which is set to ‘>=’ for continuous and ‘==’ for categorical values respectively). 

Best feature  
Our evaluation function is the gini index which can be used to decide the best features. The lower the gini index, the better the split. So features with least gini can be considered to be the best features. 

Post-processing 

There is no such post-processing required. We just use cross validation.
