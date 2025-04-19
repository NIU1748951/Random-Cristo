from collections import Counter
import numpy as np
from numpy.typing import NDArray

from .DecisionTree import Leaf, Parent
from .Dataset import Dataset
from utils.LoggerConfig import get_logger

logger =  get_logger(__name__)

class RandomForestClassifier:
    def __init__(self, max_depth: int, min_size_split: int, ratio_samples: float,
                 num_trees: int, num_random_features: int, criterion: str):
        self._max_depth = max_depth
        self._min_size_split = min_size_split
        self._ratio_samples = ratio_samples
        self._num_trees = num_trees
        self._num_random_features = num_random_features
        self._criterion = criterion
        logger.info(f"""Initializing RandomForestClassifier.
- Maximum depth: {max_depth}
- Minimum size split: {min_size_split}
- Ratio samples: {ratio_samples}
- Number of trees: {num_trees}
- Number of random features: {num_random_features}
- Criterion selected: {criterion}""")

    def predict(self, features: NDArray):
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
    
    def fit(self, features: NDArray, labels: NDArray):
        logger.info("Starting fitting process...")
        dataset = Dataset(features, labels)
        self._make_decision_trees(dataset)

    def _make_decision_trees(self, dataset: Dataset):
        self._trees = []
        for i in range(self._num_trees):
            # bootstrap
            subset = dataset.random_sampling(self._ratio_samples, True)
            tree = self._make_node(subset, 1) # The root
            self._trees.append(tree)
            logger.debug(f"Appended tree #{i + 1}")

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

        left_gini = self._gini_impurity(left_dataset.labels)
        right_gini = self._gini_impurity(right_dataset.labels)
        
        return left_weight * left_gini + right_weight * right_gini

    def _gini_impurity(self, labels: NDArray) -> float:
        # We should use Strategy pattern!!!
        
        if len(labels) == 0:
            return 0

        counter = Counter(labels)
        proportions = [count/len(labels) for count in counter.values()]

        # G(D) = 1 - Σ p_c²
        return 1 - sum(p**2 for p in proportions)


