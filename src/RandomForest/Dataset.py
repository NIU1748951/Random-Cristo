#int 64
from math import floor
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray

import utils

logger = utils.get_logger(__name__)

#NOTE: Our Docstrings follow the PEP257 standards 

def _cast_features(col) -> np.ndarray:
    """Tries to cast feature column to optimal numerical type.
    
    Tries casting priorities: int > float > string
    
    Args:
        col: Input column of feature values
        
    Returns:
        np.ndarray: Array cast to most appropriate dtype
        
    Note:
        Maintains string values if numerical casting fails
    """
    try:
        return np.array(col, dtype=int)
    except ValueError:
        try:
            return np.array(col, dtype=float)
        except:
            return np.array(col, dtype=str)

def _categorize(labels):
    """Convert class labels to integer representations.
    
    Args:
        labels: Iterable of class labels (strings or numbers)
        
    Returns:
        np.ndarray: Integer-encoded labels
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


@dataclass
class Dataset:
    """IMMUTABLE dataset container for machine learning tasks.
    
    Handles data loading, preprocessing, and common dataset operations.
    
    Attributes:
        features: 2D array of input features (n_samples, n_features)
        labels: 1D array of target labels (n_samples,)
    """
    __slots__ = ["features", "labels"]
    features: NDArray[Any]
    labels: NDArray[Any]
    
    @classmethod
    def from_file(cls, filename: str, separator: str = ','):
        """Create Dataset from CSV-like file.
        
        Args:
            filename: Path to data file
            separator: Column delimiter character
            
        Returns:
            Dataset: Loaded and preprocessed dataset
            
        Raises:
            ValueError: If file has inconsistent column counts
        """
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

    @property
    def mean_value(self):
        return np.mean(self.labels)

    def train_test_split(self, test_size: float = 0.25, shuffle: bool = True) -> Tuple["Dataset", "Dataset"]:
        """Split dataset into training and testing subsets.
        
        Args:
            test_size: Proportion of dataset to allocate for testing (0-1)
            shuffle: Randomize sample order before splitting
            
        Returns:
            Tuple[Dataset, Dataset]: (train_set, test_set)
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
        """Create bootstrap sample from dataset.
        
        Args:
            ratio: Proportion of samples to select (0-1)
            replace: Allow duplicate samples (with replacement)
            
        Returns:
            Dataset: Sampled subset
        """
        assert 0.0 < ratio <= 1.0  # It's a percentage!
        size = floor(self.n_samples * ratio)
        assert size > 0.0
        idx = np.random.choice(range(self.n_samples), size, replace)
        return Dataset(self.features[idx], self.labels[idx])
    
    def split(self, index_feature: int, threshold: float) -> Tuple["Dataset", "Dataset"]:
        """Partition dataset based on feature threshold.
        
        Args:
            index_feature: Feature column to split on
            threshold: Value threshold for splitting
            
        Returns:
            Tuple[Dataset, Dataset]: (left_split, right_split), in which:
                left_split: Samples with feature <= threshold
                right_split: Samples with feature > threshold
        """
        idx_left = np.ndarray = self.features[:, index_feature] < threshold
        idx_right = np.ndarray = self.features[:, index_feature] >= threshold
        return Dataset(self.features[idx_left], self.labels[idx_left]), Dataset(self.features[idx_right], self.labels[idx_right])

