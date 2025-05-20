from multiprocessing import Pool
import os
from abc import ABC
from typing import override, Any

import tqdm
import numpy as np
from numpy.typing import NDArray

from .DecisionTree import Leaf, Parent
from .Dataset import Dataset
from .Strategy import*
from utils.LoggerConfig import get_logger

logger = get_logger(__name__)

#NOTE: Our Docstrings follow the PEP257 standards 

class RandomForest(ABC):
    def __init__(self, max_depth: int, min_size_split: int, ratio_samples: float,
                 num_trees: int, num_random_features: int, criterion: str | ImpurityStrategy = "gini", 
                 *, n_jobs: int | None = None):
        """
        The main random forest classifier.
    
        Implements the random forest algorithm with support for:
        - Two different impurity criteria (Gini, Entropy)
        - Parallel tree construction
        - Bootstrap sampling of data and features
        - Customizable tree depth and splitting rules
        
        Args:
            max_depth: Maximum allowed depth for individual decision trees
            min_size_split: Minimum number of samples required to split a node
            ratio_samples: Ratio of samples to use in bootstrap sampling
            num_trees: Number of decision trees in the forest
            num_random_features: Number of features to consider at each split
            criterion: Impurity calculation strategy (string or ImpurityStrategy instance)
            n_jobs: Number of parallel jobs for tree construction (-1 = use all cores)
        
        Attributes:
            _trees (list): Collection of decision trees comprising the forest
            _impurity_strat (ImpurityStrategy): Current impurity calculation strategy

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


    def set_impurity_strategy(self, strategy: ImpurityStrategy | str):
        """Update of  the impurity calculation strategy used by the classifier.
        
        Args:
            strategy: Either an ImpurityStrategy instance or a string identifier
        
        Raises:
            Warning: If an unrecognized string identifier is provided
        """

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
        """Mapping of a string identifier to a concrete ImpurityStrategy instance.
        
        Args:
            criterion: Impurity criterion identifier ('gini' or 'entropy')
        
        Returns:
            Corresponding ImpurityStrategy instance (None if unrecognized)
        """
        match criterion:
            case "gini":
                return ImpurityStrategyGini()
            case "entropy":
                return ImpurityStrategyEntropy()
            case "sse" | "sumsquarederrors" | "sum_squared_errors":
                return ImpurityStrategySSE()
            case _:
                return None

    def predict(self, X: NDArray | Dataset) -> NDArray:
        """
        Predict results from a given dataset or features array

        Args:
            X: Input data(Dataset instance)

        Returns:
            NDArray: Predicted values for all samples

        Raises:
            ValueError if called before model fitting
        """
        if isinstance(X, Dataset):
            features = X.features
        elif isinstance(X, np.ndarray):
            features = X
        else:
            raise ValueError("Invalid arguments for predicting")

        if len(self._trees) == 0:
            raise ValueError("Model must be trained before predicting!")

        y_pred = []
        for f in features:
            predictions = [tree.predict(f) for tree in self._trees]
            y_pred.append(self._combine_predictions(predictions))
        return np.array(y_pred)

    @abstractmethod
    def _combine_predictions(self, predictions):
        pass

    def fit(self, X: NDArray | Dataset, y: NDArray | None = None):
        """Trains the random forest model on input data.
        
        Args:
            X: Training data (Dataset instance)
            y: Target labels 
        
        Raises:
            ValueError: For invalid input arguments
        """
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
        """Decides the decision tree construction process.
        
        Selects between sequential and parallel execution based on n_jobs parameter, explained above.
        
        Args:
            dataset: Complete training dataset
        """
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
        for i in tqdm.tqdm(range(self._num_trees), desc="Creating decision trees", ascii=False, ncols=100):
            # bootstrap
            subset = dataset.random_sampling(self._ratio_samples, True)
            tree = self._make_node(subset, 1) # The root
            self._trees.append(tree)
            logger.debug(f"Appended tree #{i + 1}")

    def _fit_target(self, dataset: Dataset):
        subset = dataset.random_sampling(self._ratio_samples, True)
        tree = self._make_node(subset, 1)
        logger.debug(f"Created tree {os.getpid()}")
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
        self._trees = []

        with Pool(processes=n_workers, initializer=self._initial_worker, initargs=(self, dataset)) as pool:
            for tree in tqdm.tqdm(pool.imap(self._worker, tasks),
                                  total=self._num_trees, 
                                  desc="Creating decision trees"):
                self._trees.append(tree)

        logger.info("Finished fitting process")

    def _make_node(self, dataset: Dataset, depth: int):
        """Recursively constructs a decision tree node.
        
        Args:
            dataset: Current subset of training data
            depth: Current depth in the tree structure
            
        Returns:
            Node: Leaf node if stopping conditions met, else returns an internal decision node
        """
        if depth >= self._max_depth \
            or dataset.n_samples <= self._min_size_split\
            or len(np.unique(dataset.labels)) <= 1:
            node = self._make_leaf(dataset)
        else:
            node = self._make_parent_or_leaf(dataset, depth)

        return node

    @abstractmethod
    def _make_leaf(self, dataset: Dataset) ->Leaf:
        pass

    def _make_parent_or_leaf(self, dataset: Dataset, depth: int):
        """Attempts to create a decision node (Parent), falls back to leaf if split is invalid.
    
        This is the core tree-building method that:
        1.-Randomly selects features for splitting
        2.-Finds the best feature/threshold combination
        3.-Validates split viability
        4.-Recursively builds child nodes or creates leaf
        
        Args:
            dataset: Current data partition being processed
            depth: Current depth in the tree hierarchy
            
        Returns:
            Parent: If valid split found and children created
            Leaf: If split produces empty partitions or other stopping conditions
        
        Note:
            Even after finding a split, may still return leaf if child nodes
            would contain zero samples (prevoves invalid trees)
        """
        idx_features = np.random.choice(range(dataset.n_features), self._num_random_features, replace=True)
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
        """Finds optimal feature split using the CART algorithm.
        
        Args:
            idx_features: Candidate feature indices for splitting
            dataset: Current data subset
            
        Returns:
            A Tuple: (best_feature_index, best_threshold, min_cost, (left_split, right_split))
        """
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
        """Calculates weighted impurity cost for a potential split.
        
        Args:
            left_dataset: Left child data after split
            right_dataset: Right child data after split
            
        Returns:
            float: Weighted sum of child node impurities
        """
        # Compute J(k, v) equation
        total_samples = left_dataset.n_samples + right_dataset.n_samples

        left_weight = left_dataset.n_samples / total_samples
        right_weight = right_dataset.n_samples / total_samples

        assert  self._impurity_strat != None # It should never be equal to None!
        left_gini = self._impurity_strat.execute(left_dataset.labels)
        right_gini = self._impurity_strat.execute(right_dataset.labels)
        
        return left_weight * left_gini + right_weight * right_gini
    
    def apply_visitor(self, visitor):
        """Apply a visitor to all trees in the forest."""
        for tree in self._trees:
            tree.accept(visitor)

class RandomForestClassifier(RandomForest):
    def __init__(self, max_depth: int, min_size_split: int, ratio_samples: float, num_trees: int,
                 num_random_features: int, criterion: str | ImpurityStrategy = "gini",
                 *, n_jobs: int | None = None):
        super().__init__(max_depth, min_size_split, ratio_samples, num_trees, num_random_features, criterion, n_jobs=n_jobs)
        logger.info(f"""Initializing RandomForestClassifier.
- Maximum depth: {max_depth}
- Minimum size split: {min_size_split}
- Ratio samples: {ratio_samples}
- Number of trees: {num_trees}
- Number of random features: {num_random_features}
- Impurity measure algorithm: {self._impurity_strat}""")

    @override
    def _combine_predictions(self, predictions) -> Any:
        return np.argmax(np.bincount(predictions))

    @override
    def _make_leaf(self, dataset: Dataset):
        return Leaf(int(dataset.most_freq_label))

class RandomForestRegressor(RandomForest):
    def __init__(self, max_depth: int, min_size_split: int, ratio_samples: float, num_trees: int,
                 num_random_features: int, criterion: str | ImpurityStrategy = "sse",
                 *, n_jobs: int | None = None):
        super().__init__(max_depth, min_size_split, ratio_samples, num_trees, num_random_features, criterion, n_jobs=n_jobs)
        logger.info(f"""Initializing RandomForestRegressor.
- Maximum depth: {max_depth}
- Minimum size split: {min_size_split}
- Ratio samples: {ratio_samples}
- Number of trees: {num_trees}
- Number of random features: {num_random_features}
- Impurity measure algorithm: {self._impurity_strat}""")

    @override
    def _combine_predictions(self, predictions):
        return np.mean(predictions)

    @override
    def _make_leaf(self, dataset: Dataset):
        return Leaf(float(dataset.mean_value))
