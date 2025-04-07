import numpy as np
from typing import Any, Tuple, List, Optional
from collections import Counter

from .DecisionTree import Node, LeafNode, DecisionNode
from .Dataset import Dataset

class RandomForestClassifier():
    def __init__(self, max_depth: int, min_size_split: int, ratio_samples: float,
                 num_trees: int, num_random_features: int, criterion: str):
        self._max_depth = max_depth
        self._min_size_split = min_size_split
        self._ratio_samples = ratio_samples
        self._num_trees = num_trees
        self._num_random_features = num_random_features
        self._criterion = criterion
        self._trees: List[Node] = []

    def fit(self, X, y):
        dataset = Dataset(X, y)
        self._build_trees(dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(self._trees) == 0:
            raise ValueError("Model must be trained before predicting!")

        n_samples = X.shape[0]
        predictions = np.empty(n_samples, dtype=object)
        
        for i in range(n_samples):
            votes = []
            for tree in self._trees:
                votes.append(tree.predict(X[i]))
            
            most_common = Counter(votes).most_common(1)[0][0]
            predictions[i] = most_common

        return predictions

    def _build_trees(self, dataset: Dataset):
        self._trees = []
        for _ in range(self._num_trees):
            # Create each tree
            subset: Any = dataset.random_sample(self._ratio_samples)
            tree = self._make_node(subset, depth=1) # This will be the root of the tree
            self._trees.append(tree)

    def _make_node(self, dataset: Dataset, depth: int) -> Node:
        if (depth >= self._max_depth or                     # if we already have maximum node depth...
            dataset.num_samples <= self._min_size_split or  # or if we have too few samples...
            len(np.unique(dataset.y)) == 1):                # or if all samples brlong to the same class
            
            node = self._make_leaf(dataset)
        else:
            node = self._split_node(dataset, depth)

        return node

    def _make_leaf(self, dataset: Dataset):
        return LeafNode(dataset.most_frequent_label())

    def _split_node(self, dataset: Dataset, depth: int):
        # Select random features (we get the indices)
        idx_features = np.random.choice(range(dataset.num_features), self._num_random_features,
                                       replace=False)

        best_feature_index, best_threshold, best_split = self._best_split(idx_features, dataset)
        if best_split is None:
            return self._make_leaf(dataset)

        left_dataset, right_dataset = best_split
        if left_dataset.num_samples == 0 or right_dataset.num_samples == 0:
            return self._make_leaf(dataset)
        else:
            node = DecisionNode(best_feature_index, best_threshold, 
                                self._make_node(left_dataset, depth + 1),
                                self._make_node(right_dataset, depth + 1))
            return node

    def _best_split(self, idx_features: np.ndarray, dataset: Dataset) -> Tuple[int | float, int | float, Optional[List[Dataset]]]:
        # Find best pairs (feature, threshold) by exploring all possible pairs
        best_feature_index, best_threshold, minimum_cost, best_split = np.inf, np.inf, np.inf, None
        for idx in idx_features:
            values = np.unique(dataset.X[:, idx])
            for val in values:
                left_dataset, right_dataset = dataset.split(idx, val)
                assert left_dataset != None and right_dataset != None
                cost = self._CART_cost(left_dataset, right_dataset) # J(k,v)
                if cost < minimum_cost:
                    best_feature_index, best_threshold, minimum_cost, best_split = \
                            idx, val, cost, [left_dataset, right_dataset]
        return best_feature_index, best_threshold, best_split

    def _CART_cost(self, left_dataset: Dataset, right_dataset: Dataset) -> float:
        # Compute J(k, v) equation
        total_samples = left_dataset.num_samples + right_dataset.num_samples

        left_weight = left_dataset.num_samples / total_samples
        right_weight = right_dataset.num_samples / total_samples

        left_gini = self._gini_impurity(left_dataset.y)
        right_gini = self._gini_impurity(right_dataset.y)
        
        return left_weight * left_gini + right_weight * right_gini

    def _gini_impurity(self, y: np.ndarray) -> float:
        # We should use Strategy pattern!!!
        
        if len(y) == 0:
            return 0

        counter = Counter(y)
        proportions = [count/len(y) for count in counter.values()]

        # G(D) = 1 - Σ p_c²
        return 1 - sum(p**2 for p in proportions)


