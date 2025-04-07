from abc import ABC, abstractmethod
from .Dataset import Dataset

class Node(ABC):
    @abstractmethod
    def predict(self, X):
        pass

# Represents the final prediction
class LeafNode(Node):
    def __init__(self, value):
        self._value = value

    def predict(self, X):
        return self._value

class DecisionNode(Node):
    def __init__(self, feature_index, threshold, left: Node, right: Node):
        self._feature_index = feature_index
        self._threshold = threshold
        self._left = left       # meets the condition
        self._right = right     # doesn't meet the condition

    def predict(self, X):
        if X[self._feature_index] <= self._threshold:
            return self._left.predict(X)
        else:
            return self._right.predict(X)

