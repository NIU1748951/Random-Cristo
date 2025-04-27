from collections import Counter
from multiprocessing import Pool
import os

import numpy as np
from numpy.typing import NDArray

from .DecisionTree import Leaf, Parent
from .Dataset import Dataset
from .Strategy import*
from utils.LoggerConfig import get_logger

logger = get_logger(__name__)

class RandomForestClassifier:
    def __init__(self, max_depth: int, min_size_split: int, ratio_samples: float,
                 num_trees: int, num_random_features: int, criterion: str | ImpurityStrategy = "gini", 
                 *, n_jobs: int | None = None):
        """
        N_jobs follows same convention as scikit-learn:
            - n_jobs = None   -> Same behaviour as n_jobs = 1 (sequential)
            - n_jobs = 1      -> Sequential
            - n_jobs > 1      -> Use this number of threads
            - n_jobs = -1     -> Use all available cores
        """
        self._max_depth = max_depth
        self._min_size_split = min_size_split
        self._ratio_samples = ratio_samples
        self._num_trees = num_trees
        self._num_random_features = num_random_features
        self._n_jobs = n_jobs

        if isinstance(criterion, str):
            self._impurity_strat = self._match_criterion(criterion.lower())
        elif isinstance(criterion, ImpurityStrategy):
            self._impurity_strat = criterion

        if self._impurity_strat == None:
            logger.warning(f"Impurity algorithm '{criterion}' is unkown. Selecting default algorithm 'GINI'")
            self._impurity_strat = ImpurityStrategyGini()

        logger.info(f"""Initializing RandomForestClassifier.
- Maximum depth: {max_depth}
- Minimum size split: {min_size_split}
- Ratio samples: {ratio_samples}
- Number of trees: {num_trees}
- Number of random features: {num_random_features}
- Impurity measure algorithm: {self._impurity_strat}""")

    def set_impurity_strategy(self, strategy: ImpurityStrategy | str):
        if isinstance(strategy, ImpurityStrategy):
            self._impurity_strat = strategy
            logger.info(f"Impurity method changed to '{strategy}'")
        else:
            new_strategy = self._match_criterion(strategy)
            if new_strategy == None:
                logger.warning(f"Impurity algorithm '{new_strategy}' is unkown. No changes were made")
            else:
                self._impurity_strat = new_strategy
                logger.info(f"Impurity method changed to '{strategy}'")

    @staticmethod
    def _match_criterion(criterion: str) -> ImpurityStrategy | None:
        match criterion:
            case "gini":
                return ImpurityStrategyGini()
            case "entropy":
                return ImpurityStrategyEntropy()
            case _:
                return None

    def predict(self, X: NDArray | Dataset):
        if isinstance(X, Dataset):
            features = X.features
        elif isinstance(X, np.ndarray):
            features = X
        else:
            raise ValueError("Invalid arguments for predicting")

        if len(self._trees) == 0:
            raise ValueError("Model must be trained before predicting!")

        n_samples = features.shape[0]
        predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            votes = []
            for tree in self._trees:
                votes.append(tree.predict(features[i]))
            
            most_common = Counter(votes).most_common(1)[0][0]
            predictions[i] = most_common

        return predictions

    def fit(self, X: NDArray | Dataset, y: NDArray | None = None):
        if isinstance(X, Dataset):
            features = X.features
            labels = X.labels
        elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            features = X
            labels = y
        else:
            raise ValueError("Invalid arguments for training")

        logger.info("Starting fitting process...")
        dataset = Dataset(features, labels)
        self._make_decision_trees(dataset)

    def _make_decision_trees(self, dataset: Dataset):
        if self._n_jobs == -1:
            num_workers = os.cpu_count()
        else:
            num_workers = self._n_jobs

        if num_workers is None or num_workers <= 1:
            self._make_decision_trees_sequential(dataset)
        else:
            self._make_decision_trees_parallel(dataset, num_workers)


    def _make_decision_trees_sequential(self, dataset: Dataset):
        logger.info("Creating decision trees in 'sequential' mode")
        self._trees = []
        for i in range(self._num_trees):
            # bootstrap
            subset = dataset.random_sampling(self._ratio_samples, True)
            tree = self._make_node(subset, 1) # The root
            self._trees.append(tree)
            logger.debug(f"Appended tree #{i + 1}")

    def _fit_target(self, dataset: Dataset):
        subset = dataset.random_sampling(self._ratio_samples, True)
        tree = self._make_node(subset, 1)
        return tree

    @staticmethod
    def _initial_worker(rf, dataset):
        """
        Sets a helping global variable containing (self, ) a.k.a the Random Forest mode
        to use in the actual workers
        """
        global _workers_rf, _workers_dataset
        _workers_rf = rf
        _workers_dataset = dataset

    @staticmethod
    def _worker(_):
        """
        Pickable function to be passed to `pool.map`
        The argument '_' needs to be defined but is actually unused
        """
        return _workers_rf._fit_target(_workers_dataset)
    
    def _make_decision_trees_parallel(self, dataset: Dataset, n_workers: int):
        logger.info(f"Creating decision trees in 'parallel' mode with {n_workers} workers")
        tasks = [None] * self._num_trees

        with Pool(processes=n_workers, initializer=self._initial_worker, initargs=(self, dataset)) as pool:
            self._trees = pool.map(self._worker, tasks)


        logger.info("Finished fitting process")

    def _make_node(self, dataset: Dataset, depth: int):
        if depth >= self._max_depth \
            or dataset.n_samples <= self._min_size_split\
            or len(np.unique(dataset.labels)) <= 1:
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)

        return node

    def _make_leaf(self, dataset: Dataset):
        return Leaf(int(dataset.most_freq_label))

    def _make_parent_or_leaf(self, dataset: Dataset, depth: int):
        idx_features = np.random.choice(range(dataset.n_features), self._num_random_features, replace=False)
        best_feature_idx, best_threshold, _, best_split = self._best_split(idx_features, dataset)

        left_dataset, right_dataset = best_split
        assert left_dataset is not None and right_dataset is not None
        assert left_dataset.n_samples > 0 or right_dataset.n_samples > 0
        if left_dataset.n_samples == 0 or right_dataset.n_samples == 0:
            return self._make_leaf(dataset)
        else:
            left_child = self._make_node(left_dataset, depth + 1)
            right_child = self._make_node(right_dataset, depth + 1)
            node = Parent(int(best_feature_idx), best_threshold, left_child, right_child)
            return node

    def _best_split(self, idx_features: NDArray, dataset: Dataset):
        best_feature_index, best_threshold, minimum_cost, best_split = \
            np.inf, np.inf, np.inf, (None, None)

        for idx in idx_features:
            values = np.unique(dataset.features[:, idx])
            for value in values:
                left_dataset, right_dataset = dataset.split(idx, value)
                cost = self._CART_cost(left_dataset, right_dataset)
                if cost < minimum_cost:
                    best_feature_index = idx
                    best_threshold = value
                    minimum_cost = cost
                    best_split = (left_dataset, right_dataset)

        return best_feature_index, best_threshold, minimum_cost, best_split

    def _CART_cost(self, left_dataset: Dataset, right_dataset: Dataset) -> float:
        # Compute J(k, v) equation
        total_samples = left_dataset.n_samples + right_dataset.n_samples

        left_weight = left_dataset.n_samples / total_samples
        right_weight = right_dataset.n_samples / total_samples

        assert  self._impurity_strat != None # It should never be equal to None!
        left_gini = self._impurity_strat.execute(left_dataset.labels)
        right_gini = self._impurity_strat.execute(right_dataset.labels)
        
        return left_weight * left_gini + right_weight * right_gini

