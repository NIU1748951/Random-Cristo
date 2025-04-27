#int 64
from math import floor
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

import utils

logger = utils.get_logger(__name__)

def _cast_features(col) -> np.ndarray:
    """
    Tries to cast each feature into an int or a float and if it cant it leaves it as a string
    """
    try:
        return np.array(col, dtype=int)
    except ValueError:
        try:
            return np.array(col, dtype=float)
        except:
            return np.array(col, dtype=str)

def _categorize(labels):
    """
    Forces each label to be an integer representing its class
    """
    try:
        return np.array(labels, dtype=int)
    except ValueError:
        # It must be a string that represents its class
        categories = []
        labels_int = np.empty(len(labels), dtype=int)
        for j,cat in enumerate(labels):
            if cat not in categories:
                categories.append(cat)
            labels_int[j] = categories.index(cat)
        return labels_int


@dataclass(frozen=True)
class Dataset:
    __slots__ = ["features", "labels"]
    features: NDArray[Any]
    labels: NDArray[Any]
    
    @classmethod
    def from_file(cls, filename: str, separator: str = ','):
        features = []
        labels = []
        n_fields = None

        with open(filename, "r") as file:
            for line in file:
                parts = line.strip().split(separator)
                if n_fields is None:
                    n_fields = len(parts)
                elif len(parts) != n_fields:
                    raise ValueError("File is not well formatted")

                features.append(parts[:-1])
                labels.append(parts[-1])

        # Transpose
        features_t = list(zip(*features))
        casted_cols = [_cast_features(col) for col in features_t]

        try:
            features_arr = np.stack(casted_cols, axis=1)
        except ValueError:
            features_arr = np.empty((len(features), len(casted_cols)), dtype=object)
            for j, col in enumerate(casted_cols):
                features_arr[:, j] = col

        labels_arr = _categorize(labels)

        logger.info(f"Created dataset with {n_fields} fields and {features_arr.shape[0]} samples")
        logger.info(f"Features:\n{features_arr}")
        logger.info(f"Labels:\n{labels_arr}")
        return cls(features_arr, labels_arr)

    @property
    def n_features(self) -> int:
        return self.features.shape[1]
    
    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    @property
    def most_freq_label(self) -> np.intp:
        return np.argmax(np.bincount(self.labels))

    def train_test_split(self, test_size: float = 0.25, shuffle: bool = True) -> Tuple["Dataset", "Dataset"]:
        """
        Separates the dataset in two: train, test
        """

        assert 0 < test_size <= 1, "Variable 'test_size' must be a percentage"
        idx = np.arange(self.n_samples)
        if shuffle:
            rng = np.random.default_rng(None)
            rng.shuffle(idx)

        split_idx = int(self.n_samples * (1 - test_size))
        train_idx, test_idx = idx[:split_idx], idx[split_idx:]

        train = Dataset(self.features[train_idx], self.labels[train_idx])
        test = Dataset(self.features[test_idx], self.labels[test_idx])
        return train, test

    def random_sampling(self, ratio: float, replace: bool = True) -> "Dataset":
        assert 0.0 < ratio <= 1.0  # It's a percentage!
        size = floor(self.n_samples * ratio)
        assert size > 0.0
        idx = np.random.choice(range(self.n_samples), size, replace)
        return Dataset(self.features[idx], self.labels[idx])
    
    def split(self, index_feature: int, threshold: float) -> Tuple["Dataset", "Dataset"]:
        idx_left = np.ndarray = self.features[:, index_feature] < threshold
        idx_right = np.ndarray = self.features[:, index_feature] >= threshold
        return Dataset(self.features[idx_left], self.labels[idx_left]), Dataset(self.features[idx_right], self.labels[idx_right])

