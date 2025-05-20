from typing import Any

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from RandomForest.TreeVisitor import FeatureImportanceVisitor, TreePrinterVisitor

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

def load_daily_min_temperatures():
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/'
    'Datasets/master/daily-min-temperatures.csv')
    # Minimum Daily Temperatures Dataset over 10 years (1981-1990)
    # in Melbourne, Australia. The units are in degrees Celsius.
    # These are the features to regress:
    day = pd.DatetimeIndex(df.Date).day.to_numpy() # 1...31 #type: ignore
    month = pd.DatetimeIndex(df.Date).month.to_numpy() # 1...12 #type: ignore
    year = pd.DatetimeIndex(df.Date).year.to_numpy() # 1981...1999 #type: ignore
    X = np.vstack([day, month, year]).T # np array of 3 column #type: ignore
    y = df.Temp.to_numpy()
    return X, y

def RMSE(y_pred, y_test):
    return np.sqrt(np.mean((y_pred - y_test) ** 2))


def main():
    #iris: Any = load_wine()
    #X, y = iris.data, iris.target
    #dataset = model.Dataset.from_file("data/Credit_card/creditcard_10K.csv") 
    #X, y = dataset.features, dataset.labels

    X, y = load_daily_min_temperatures()

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
    max_depth = 16
    min_size_split = 8  # if less, do not split the node
    ratio_samples = 0.7 # sampling with replacement
    num_trees = 1000
    num_random_features = int(np.sqrt(num_features))
                        # number of features to consider at each node
                        # when looking for the best split

    rf = model.RandomForestRegressor(max_depth, min_size_split, ratio_samples,
                                      num_trees, num_random_features, "sse",
                                      n_jobs=-1)

    rf.fit(X, y)

    y_pred = rf.predict(X_test)


    fi_visitor = FeatureImportanceVisitor()
    rf.apply_visitor(fi_visitor)
    importances = fi_visitor.get_importances()
    logger.info("Feature importances:")
    for idx, score in sorted(importances.items()):
        logger.info(f"Feature {idx}: {score:.4f}")


    logger.info("Sample tree structure:")
    printer = TreePrinterVisitor()
    rf._trees[0].accept(printer)

    #num_samples_test = len(y_test)
    #num_correct_predictions = np.sum(ypred == y_test)
    #accuracy = num_correct_predictions / float(num_samples_test)

    #logger.info("Accuracy: {} %".format(100 * np.round(accuracy, decimals=2)))
    #if accuracy < 0.5:
    #    logger.warning("Accuracy is too low!")

    error = RMSE(y_pred, y_test)
    logger.info("Error: {}".format(np.round(error, decimals=2)))
    if error > 50.0:
        logger.warning("Accuracy is too low!")

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        logger.error("Unimplemented error: " + str(e))
    except ValueError as e:
        logger.error("Value error occured: ", str(e))

