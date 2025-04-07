from typing import Any
import numpy as np
import sklearn.datasets

import RandomForest as model

def main():
    iris: Any = sklearn.datasets.load_iris()
    X, y = iris.data, iris.target
    ratio_train, ratio_test = 0.7, 0.3  
    # 70% train, 30% test
    num_samples, num_features = X.shape
    # num_samples is the number of rows
    # num_features is the number of columns
    idx = np.random.permutation(range(num_samples))
    # returns a list with a randomized series of indexs

    num_samples_train = int(num_samples * ratio_train)
    num_samples_test = int(num_samples * ratio_test)
    idx_train = idx[:num_samples_train]
    idx_test = idx[num_samples_train : num_samples_train + num_samples_test]
    X_train, y_train = X[idx_train], y[idx_train]
    X_test, y_test = X[idx_test], y[idx_test]

    # Hyperparameters
    max_depth = 10 
    min_size_split = 5  # if less, do not split the node
    ratio_samples = 0.7  # sampling with replacement
    num_trees = 10
    num_random_features = int(np.sqrt(num_features))
                        # number of features to consider at each node
                        # when looking for the best split
    criterion = "gini"

    rf = model.RandomForestClassifier(max_depth, min_size_split, ratio_samples,
                                      num_trees, num_random_features, criterion)

    rf.fit(X_train, y_train)
    ypred = rf.predict(X_test)

    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions / float(num_samples_test)
    print("Accuracy: {} %".format(100 * np.round(accuracy, decimals=2)))


def test():
    left = model.LeafNode(0)
    right = model.LeafNode(1)

    root = DecisionNode(feature_index=0, )

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        print("Unimplemented error: " + str(e))

