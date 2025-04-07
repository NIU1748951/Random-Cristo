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
        if not (0 < ratio_samples <= 1):
            raise ValueError("ratio_samples must be between 0 and 1")
        
        samples_to_take = int(self.num_samples * ratio_samples)

        if samples_to_take == 0:
            return Dataset(np.empty((0, self.num_features)), np.empty(0))

        random_indices = np.random.choice(
                range(self.num_samples),
                size=samples_to_take,
                replace=False
        )
        
        return Dataset(
            X=self.X[random_indices],
            y=self.y[random_indices]
        )


    def split(self, idx: int, value: int | float) -> Tuple[Optional['Dataset'], Optional['Dataset']]:
        if not (0 <= idx <= self.num_features):
            raise ValueError(f"idx must be between 0 and {self.num_features - 1}. Current value: {idx}")

        left_indices = np.where(self.X[:, idx] <= value)[0]
        right_indices = np.where(self.X[:, idx] <= value)[0]

        left_dataset = None
        right_dataset = None

        if len(left_indices) > 0:
            left_dataset = Dataset(
                X=self.X[left_indices],
                y=self.y[left_indices]
            )

        if len(right_indices) > 0:
            right_dataset = Dataset(
                X=self.X[right_indices],
                y=self.y[right_indices]
            )

        return left_dataset, right_dataset

