from math  import sqrt

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

    # Inicializar y entrenar
    rf = RandomForestClassifier(**rf_params)
    rf.fit(train)

    y_pred = rf.predict(test)
    accuracy = np.mean(y_pred == test.labels)
    return accuracy


def run_experiments(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
) -> pd.DataFrame:
    records = []
    for name, (X, y) in datasets.items():
        logger.info(f"Iniciando experimentos en dataset '{name}'")
        for run in tqdm(range(1, n_runs + 1), desc=f"{name}"):
            acc = evaluate_dataset(X, y, rf_params)
            records.append({
                'dataset': name,
                'run': run,
                'accuracy': acc
            })
    return pd.DataFrame.from_records(records)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df
        .groupby('dataset')['accuracy']
        .agg(n_runs='count', mean_accuracy='mean', std_accuracy='std')
        .reset_index()
    )
    return summary


def run_and_report(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
    test_size: float = 0.25,
    save_csv: str = "Results.csv"
) -> pd.DataFrame:
    results = run_experiments(datasets, rf_params, n_runs)
    summary = summarize_results(results)
    if save_csv:
        summary.to_csv(save_csv, index=False)
        logger.info(f"Resumen guardado en {save_csv}")
    return summary


if __name__ == '__main__':
    datasets = {
        'iris': load_iris(return_X_y=True),
        'wine': load_wine(return_X_y=True),
        'digits': load_digits(return_X_y=True)
    }

    rf_params = {
        "max_depth": 30,
        "min_size_split": 13,
        "ratio_samples": 0.7,
        "num_trees": 200,
        "num_random_features": 20,
        "criterion": "gini",
        "n_jobs": -1
    }

    summary = run_and_report(datasets, rf_params, n_runs=10, test_size=0.25,
                              save_csv='rf_summary.csv')
    print(summary)

