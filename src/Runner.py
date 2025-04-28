import numpy as np
import pandas as pd
from math  import sqrt
from sklearn.datasets import load_iris, load_wine, load_digits
from tqdm.auto import tqdm
import logging

# Import your custom RandomForest implementation
from RandomForest import RandomForestClassifier, Dataset

logger = logging.getLogger(__name__)


def evaluate_dataset(X, y, rf_params: dict, test_size: float = 0.25, random_state = None) -> float:
    """
    Entrena y evalúa una sola corrida de Random Forest sobre (X, y).

    Args:
        X (array-like): Matriz de features.
        y (array-like): Vector de etiquetas.
        rf_params (dict): Parámetros para inicializar RandomForestClassifier.
        test_size (float): Fracción de datos para test.
        random_state (int): Semilla para reproducibilidad.

    Returns:
        float: Precisión (accuracy) obtenida en el set de test.
    """
    dataset: Dataset = Dataset(X, y)
    train, test = dataset.train_test_split()

    # Inicializar y entrenar
    rf = RandomForestClassifier(13, 4, 0.7, 300, int(sqrt(train.n_features)), "gini")
    rf.fit(train)

    # Predecir y calcular accuracy
    y_pred = rf.predict(test)
    accuracy = np.mean(y_pred == test.labels)
    return accuracy


def run_experiments(
    datasets: dict,
    rf_params: dict,
    n_runs: int = 5,
    test_size: float = 0.25,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Ejecuta múltiples corridas sobre varios datasets, recogiendo precisiones.

    Args:
        datasets (dict): Mapeo de nombre_dataset -> (X, y).
        rf_params (dict): Parámetros para RandomForestClassifier.
        n_runs (int): Número de repeticiones por dataset.
        test_size (float): Fracción para test.
        random_seed (int): Semilla base.

    Returns:
        pd.DataFrame: DataFrame con columnas ['dataset', 'run', 'accuracy'].
    """
    records = []
    for name, (X, y) in datasets.items():
        logger.info(f"Iniciando experimentos en dataset '{name}'")
        for run in tqdm(range(1, n_runs + 1), desc=f"{name}"):
            seed = random_seed + run
            acc = evaluate_dataset(X, y, rf_params, test_size, random_state=seed)
            records.append({
                'dataset': name,
                'run': run,
                'accuracy': acc
            })
    return pd.DataFrame.from_records(records)


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera estadísticas agregadas de los resultados.

    Args:
        results_df (pd.DataFrame): Salida de run_experiments.

    Returns:
        pd.DataFrame: DataFrame con columnas ['dataset', 'n_runs', 'mean_accuracy', 'std_accuracy'].
    """
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
    random_seed: int = 42,
    save_csv: str = "Results.csv"
) -> pd.DataFrame:
    """
    Ejecuta experimentos y devuelve el resumen. Opcionalmente guarda CSV.
    """
    results = run_experiments(datasets, rf_params, n_runs, test_size, random_seed)
    summary = summarize_results(results)
    if save_csv:
        summary.to_csv(save_csv, index=False)
        logger.info(f"Resumen guardado en {save_csv}")
    return summary


if __name__ == '__main__':
    # Ejemplo de uso con sklearn.datasets\ n    from sklearn.datasets import load_iris, load_wine, load_digits

    datasets = {
        'iris': load_iris(return_X_y=True),
        'wine': load_wine(return_X_y=True),
        'digits': load_digits(return_X_y=True)
    }

    rf_params = {
        'max_depth': 10,
        'min_size_split': 3,
        'ratio_samples': 0.7,
        'num_random_features': None,  # se calculará como sqrt(n_features)
        'criterion': 'gini',
        'n_jobs': -1
    }

    summary = run_and_report(datasets, rf_params, n_runs=10, test_size=0.25,
                             random_seed=123, save_csv='rf_summary.csv')
    print(summary)

