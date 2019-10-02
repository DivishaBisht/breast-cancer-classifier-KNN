# breast-cancer-classifier-KNN
Classifying breast cancer using supervised machine learning algorithm KNN.

## **k-Nearest-Neighbour**
kNN is a non-parametric, lazy learning algorithm, based on feature similarity. Non-parametric meaning that kNN does not make any assumptions on the underlying data distribution. Lazy refers to the fact the kNN does not use the training data points to do any generalisation. Meaning there is no explicit training phase, which makes training very fast, but can be computationally expensive over larger training datasets. When a prediction is required for a unseen data instance, the kNN algorithm will search through the training dataset for the k most similar instances. kNN involves a similarity measure, where the distance between two data instances is calculated and used to find the k most similar neighbours


### **Parameters**
Value of k used: 5(default)
A ratio of 80/20 for train/test split of the dataset was used. This corresponds to 455 records for the training data and 114 records of unseen data.

