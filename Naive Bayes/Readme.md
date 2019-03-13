
Implementation details 
1. Import data in csv format, convert to matrix and apply KFOLD to it. 
2. K - fold splits from sklearn splits the data into train set and test set with their indexes 
3. We take train data (containing all the attributes with output) and train label (only last column of the train set). 
4. Similarly, we do for test set, test label becomes our ground truth, which we use later to measure accuracy, precision, recall and F measure using the predicted values.
5. We make a train data set dictionary, which contains the key as class labels and values as rows belonging to that class label. We use this dictionary to iterate over whole data set while testing to calculate probability distribution. 
6. For calculating of predicted values of test set  We take one test data at a time, for each column in the data being numerical or categorical our implementation goes as follows: 
  a. If data is numerical  
    i. We calculate mean and standard deviation of the train data using sklearn mean for each column belonging to one particular class. 
    ii. Similarly, we calculate standard deviation for each column of a particular class. 
    iii. Since our data is numerical, we can simply calculate probability using the probability distribution and assign class label to the one which has maximum Probability distribution value. 
  b. If data is categorical 
    i. In categorical data, we iterate through our training dictionary which contains key as label and train data as values. 
    ii. For each value of our train data belonging to one class, we check number of label classes, i.e, classes belonging to that feature or categorical value and count for given class and given attribute value , we count and store them as label_count (lC). Also, similarly those which are from same class but not belong to same attribute value, we count them as non_label _count (nlC). 
    iii. Using label and non-label count, we calculate the probability of the given attribute data value. 
    iv. And apply our result to precalculated pdf for other numerical or categorical data.
7. After every test data, we get max probability and class label it belongs to. We create a list of new test labels. 
8. Then using basic formula we calculate true positive, true negative, false positive, false negative, and with the help of which we calculate accuracy, precision, recall and F measure. 

Zero probability 
If true positive, false negative, true negative, false positive values are not zero, there is no zeroprobability condition. 
Otherwise, we handle zero probability by using Laplacian smoothing (adding 1 to the value that is 0). 
 

Parameter selection 
In Naïve Bayes, We use K Fold to split our data into train set and test set, as they are indexes , we need to need to apply few steps to convert it into train data and test data set containing whole row with all features. 
Test data is refined, by removing last row , that is class label row and then predicted against  trained data set. 

Handling continuous and categorical features 
If data is continuous 
  1. We calculate mean and standard deviation of the train data using sklearn mean for each column belonging to one particular class. 
  2. Similarly, we calculate standard deviation for each column of a particular class. 
  3. Since our data is numerical, we can simply calculate probability using the probability distribution and assign class label to the one which has maximum Probability distribution value. 
If data is categorical 
  1. For each value of our train data belonging to one class, we check number of label classes, i.e, classes belonging to that feature or categorical value and count for given class and given attribute value, we count and store them as label_count (lC). Also, similarly those which are from same class but not belong to same attribute value, we count them as non_label _count (nlC). 
  2. Using label and non-label count, we calculate the probability of the given attribute data value. 
  3. And apply our result to precalculated pdf for other numerical or categorical data. 
