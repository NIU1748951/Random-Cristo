from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import RandomForest as model
import utils 

logger = utils.get_logger(__name__)

def log_data(X, y):
    data = [(xi, yi) for xi, yi in zip(X, y)]
    num_samples, num_features = X.shape[0], X.shape[1]
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Number of features: {num_features}")
    for i, d in enumerate(data):
        logger.info(f"Sample {i}:{d[0]} - {d[1]}")

def main():
    iris: Any = load_iris()
    X, y = iris.data, iris.target

    ratio_train, ratio_test = 0.75, 0.25  
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
    min_size_split = 2  # if less, do not split the node
    ratio_samples = 0.7 # sampling with replacement
    num_trees = 300
    num_random_features = int(np.sqrt(num_features))
                        # number of features to consider at each node
                        # when looking for the best split

    rf = model.RandomForestClassifier(max_depth, min_size_split, ratio_samples,
                                      num_trees, num_random_features, "gini",
                                      n_jobs=-1)

    rf.fit(X, y)

    ypred = rf.predict(X_test)

    num_samples_test = len(y_test)
    num_correct_predictions = np.sum(ypred == y_test)
    accuracy = num_correct_predictions / float(num_samples_test)

    logger.info("Accuracy: {} %".format(100 * np.round(accuracy, decimals=2)))
    if accuracy < 0.5:
        logger.warning("Accuracy is too low!")

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        logger.error("Unimplemented error: " + str(e))
    except ValueError as e:
        logger.error("Value error occured: ", str(e))

