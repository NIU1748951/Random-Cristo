import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from tqdm.auto import tqdm

from RandomForest import RandomForestClassifier, Dataset
from utils import get_logger

logger = get_logger(__name__)

def evaluate_dataset(X, y, rf_params: dict, test_size: float = 0.25) -> float:
    dataset: Dataset = Dataset(X, y)
    train, test = dataset.train_test_split()

    rf = RandomForestClassifier(**rf_params)
    rf.fit(train)

    y_pred = rf.predict(test)
    accuracy = np.mean(y_pred == test.labels)
    return accuracy


def run_experiments_accuracy(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
) -> pd.DataFrame:
    records = []
    for name, (X, y) in datasets.items():
        logger.info(f"Starting experiments on '{name}'")
        for run in tqdm(range(1, n_runs + 1), desc=f"{name}"):
            acc = evaluate_dataset(X, y, rf_params)
            records.append({
                'dataset': name,
                'run': run,
                'accuracy': acc
            })
    return pd.DataFrame.from_records(records)

def run_experiments_time(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
) -> pd.DataFrame:
    records = []

    for name, (X, y) in datasets.items():
        logger.info(f"Starting experiments on '{name}'")
        for run in tqdm(range(1, n_runs + 1), desc=name):
            start_time = time.time()
            evaluate_dataset(X, y, rf_params)
            elapsed = time.time() - start_time
            records.append((name, run, elapsed))

    # Split tuples into per-column lists
    dataset_col = [r[0] for r in records]
    run_col = [r[1] for r in records]
    time_col = [r[2] for r in records]

    # Build DataFrame by assigning columns directly
    print(0)
    df = pd.DataFrame()
    print(1)
    df['dataset'] = dataset_col
    print(2)
    df['run'] = run_col
    print(3)
    df['time'] = time_col
    return df

def summarize_results(results_df: pd.DataFrame, param: str) -> pd.DataFrame:
    summary = (
        results_df
        .groupby('dataset')[param]
        .agg(n_runs='count', mean='mean', std='std')
        .reset_index()
    )
    return summary


def run_and_report_accuracy(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
    test_size: float = 0.25,
    save_csv: str = "Results.csv"
) -> pd.DataFrame:
    results = run_experiments_accuracy(datasets, rf_params, n_runs)
    summary = summarize_results(results, "accuracy")
    if save_csv:
        summary.to_csv(save_csv, index=False)
        logger.info(f"Summary saved on {save_csv}")
    return summary

def run_and_report_time(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
    test_size: float = 0.25,
    save_csv: str = "Results.csv",
    *,
    parallel: bool = False,
    sequential: bool = False
) -> pd.DataFrame:
    if parallel:
        rf_params["n_jobs"] = 4
    elif sequential:
        rf_params["n_jobs"] = None
    results = run_experiments_time(datasets, rf_params, n_runs)
    summary = summarize_results(results, "time")
    if save_csv:
        summary.to_csv(save_csv, index=False)
        logger.info(f"Summary saved on {save_csv}")
    return summary

if __name__ == '__main__':
    datasets = {
    #    'iris': load_iris(return_X_y=True),
    #    'wine': load_wine(return_X_y=True),
        'digits': load_digits(return_X_y=True)
    }

    rf_params = {
        "max_depth": 10,
        "min_size_split": 3,
        "ratio_samples": 0.7,
        "num_trees": 1000,
        "num_random_features": 20,
        "criterion": "gini",
        "n_jobs": -1
    }

    #summary_accuracies = run_and_report_accuracy(datasets, rf_params, n_runs=10, test_size=0.25,
    #                                             save_csv='rf_summary_acc.csv')
    #print("Summary accuracy", summary_accuracies)

    summary_par = run_and_report_time(datasets, rf_params, n_runs=10, test_size=0.25,
                              save_csv='rf_summary_parsdadasd.csv', parallel=True)
    print("Summary time parallel", summary_par)

    summary_seq = run_and_report_time(datasets, rf_params, n_runs=10, test_size=0.25,
                              save_csv='rf_summary_seq.csv', sequential=True)
    print("Summary time sequential", summary_seq)
    
