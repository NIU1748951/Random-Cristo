import numpy as np
from typing import Tuple, Optional
from collections import Counter

class Dataset():
    __slots__ = ["X", "y"]
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    @property
    def num_samples(self):
        return self.X.shape[0]

    @property
    def num_features(self):
        return self.X.shape[1]

    def most_frequent_label(self):
        return Counter(self.y).most_common(1)[0][0]

    def random_sample(self, ratio_samples: float):
        pass

    def split(self, idx: int, value: int | float) -> Tuple[Optional['Dataset'], Optional['Dataset']]:
        return None, None
