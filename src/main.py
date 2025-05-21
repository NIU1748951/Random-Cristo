from typing import Callable, Tuple, Dict

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml, load_iris, load_wine

import RandomForest as rf
import utils 

logger = utils.get_logger(__name__)

def log_data(X, y, name: str):
    data = [(xi, yi) for xi, yi in zip(X, y)]
    num_samples, num_features = X.shape[0], X.shape[1]
    logger.info(f"Showing contents of dataset {name}")
    logger.info(f"Number of samples: {num_samples}")
    logger.info(f"Number of features: {num_features}")
    for i, d in enumerate(data):
        logger.info(f"Sample {i}:{d[0]} - {d[1]}")

DatasetLoader = Callable[[], Tuple[np.ndarray, np.ndarray]]

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

__class_datasets_dict = {
    "IRIS": lambda: (load_iris(return_X_y=True)),
    "WINE": lambda: (load_wine(return_X_y=True)),
}

__regr_datasets_dict = {
    "TEMPERATURES": lambda: (load_daily_min_temperatures()),
}

def select_a_default_dataset(default_datasets: Dict) -> Tuple[rf.Dataset, str]:
    names = list(default_datasets.keys())

    print("Default Datasets:")
    for i, name in enumerate(names):
        print(f"{i + 1}) {name}")

    while True:
        print("Select an option (name or number)")
        selection = input().strip()

        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(names):
                key = names[idx]
                return rf.Dataset(*default_datasets[key]()), key #type: ignore
            else:
                print("Please, select an existing database")

        sel_upper = selection.upper()
        if sel_upper in default_datasets:
            return rf.Dataset(*default_datasets[sel_upper]()), sel_upper #type: ignore

        # If we get here then the input was not valid
        print("Please, select a valid option")

def select_a_custom_dataset() -> Tuple[rf.Dataset, str]:
    print("Custom Dataset")
    print("Please, enter the path to the Dataset file")
    filepath: str = input()
    while True:
        try:
            return rf.Dataset.from_file(filepath), filepath
        except IOError as e:
            print(f"Error: {e}. Try again.")

def select_dataset(title: str) -> Tuple[rf.Dataset, str]:
    if title not in ("CLASSIFICATOR", "REGRESSOR"):
        raise ValueError(f"Random forest type {title} is not valid")
    print(f" -- {title}  -- ")
    print("1) Use a default dataset")
    print("2) Use a custom dataset")
    while True:
        selection = None
        try:
            print("Select an option:")
            selection = int(input())
            if selection in (1, 2):
                break
            else:
                print("Please, select a valid option")
        except (ValueError, TypeError):
            print("Please, select a valid option")

    if selection == 1:
        return select_a_default_dataset(__class_datasets_dict if title == "CLASSIFICATOR" else __regr_datasets_dict)
    else:
        return select_a_custom_dataset()

def main():
    print("=====================")
    print("    RANDOM FOREST    ")
    print("=====================")
    print("                     ")
    print("Select a type of Random Forest to use: ")
    print("1) Classificator")
    print("2) Regressor")
    while True:
        try:
            selected_type = int(input("Select an option: "))
            if selected_type in (1, 2):
                break
        except (ValueError, TypeError):
            print("Please, select a valid option")

    if selected_type == 1:
        dataset, name = select_dataset("CLASSIFICATOR")
    else:
        dataset, name = select_dataset("REGRESSOR")

    print("Do you wish to print the database contents? [Y/n]")
    wants_to_see_contents = input()
    if wants_to_see_contents.lower() == "y":
        log_data(dataset.features, dataset.labels, name)

    print(f"Succesfully loaded default dataset '{name}'")

    ratio_train, ratio_test = 0.75, 0.25  
    # 70% train, 30% test

    num_samples, num_features = dataset.features.shape
    # num_samples is the number of rows
    # num_features is the number of columns

    idx = np.random.permutation(range(num_samples))
    # returns a list with a randomized series of indexs

    num_samples_train = int(num_samples * ratio_train)
    num_samples_test = int(num_samples * ratio_test)
    idx_test = idx[num_samples_train : num_samples_train + num_samples_test]
    X_test, y_test = dataset.features[idx_test], dataset.labels[idx_test]

    # Hyperparameters
    max_depth = 16
    min_size_split = 8  # if less, do not split the node
    ratio_samples = 0.7 # sampling with replacement
    num_trees = 1000
    num_random_features = int(np.sqrt(num_features))
                        # number of features to consider at each node
                        # when looking for the best split

    if selected_type == 1:
        model = rf.RandomForestClassifier(max_depth, min_size_split, ratio_samples,
                                          num_trees, num_random_features, "entropy",
                                          n_jobs=-1)
    else:
        model = rf.RandomForestRegressor(max_depth, min_size_split, ratio_samples,
                                          num_trees, num_random_features, "sse",
                                          n_jobs=-1)


    model.fit(dataset)
    y_pred = model.predict(X_test)

    if isinstance(model, rf.RandomForestClassifier):
        num_samples_test = len(y_test)
        num_correct_predictions = np.sum(y_pred == y_test)
        accuracy = num_correct_predictions / float(num_samples_test)

        logger.info("Accuracy: {} %".format(100 * np.round(accuracy, decimals=2)))
        if accuracy < 0.5:
            logger.warning("Accuracy is too low!")
    elif isinstance(model, rf.RandomForestRegressor):
        error = RMSE(y_pred, y_test)
        logger.info("Error: {}".format(np.round(error, decimals=2)))
        if error > 50.0:
            logger.warning("Accuracy is too low!")

    print("Do you want to print the decision trees on the log file? [Y/n]")
    wants_to_print = input()

    if wants_to_print.lower() == "y":
        fi_visitor = rf.FeatureImportanceVisitor()
        model.apply_visitor(fi_visitor)
        importances = fi_visitor.get_importances()
        logger.debug("Feature importances:")
        for idx, score in sorted(importances.items()):
            logger.debug(f"Feature {idx}: {score:.4f}")


        logger.info("Sample tree structure:")
        printer = rf.TreePrinterVisitor()
        model._trees[0].accept(printer)

    print(f"The Random Forest model has run succesfully!")
    print("You can check model stats and results under the logs/app.log file")

if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        logger.error("Unimplemented error: " + str(e))
    except ValueError as e:
        logger.error("Value error occured: ", str(e))

