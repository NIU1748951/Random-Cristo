from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

import utils

logger = utils.get_logger(__name__)

@dataclass(frozen=True)
class Dataset:
    __slots__ = ["features", "labels"]
    features: NDArray[np.float64]
    labels: NDArray[np.int16]
    
    @property
    def n_features(self) -> int:
        return self.features.shape[1]
    
    @property
    def n_samples(self) -> int:
        return self.features.shape[0]

    @property
    def most_freq_label(self) -> np.intp:
        return np.argmax(np.bincount(self.labels))

    def random_sampling(self, ratio: float, replace: bool = True) -> "Dataset":
        assert 0.0 < ratio <= 1.0
        size = int(self.n_samples * ratio)
        assert size > 0.0
        idx = np.random.choice(range(self.n_samples), size, replace)
        return Dataset(self.features[idx], self.labels[idx])
    
    def split(self, index_feature: int, threshold: float) -> Tuple["Dataset", "Dataset"]:
        idx_left = np.ndarray = self.features[:, index_feature] < threshold
        idx_right = np.ndarray = self.features[:, index_feature] >= threshold
        return Dataset(self.features[idx_left], self.labels[idx_left]), Dataset(self.features[idx_right], self.labels[idx_right])

