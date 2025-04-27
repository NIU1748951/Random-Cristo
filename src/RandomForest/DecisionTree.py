from dataclasses import dataclass
from typing import override
from abc import ABC, abstractmethod
from numpy.typing import NDArray

class Node(ABC):
    @abstractmethod
    def predict(self, features: NDArray):
        pass

@dataclass
class Leaf(Node):
    __slots__ = ["label"] 
    label: int

    @override
    def predict(self, _: NDArray):
        return self.label

@dataclass
class Parent(Node):
    __slots__ = ["feature_index", "threshold", "left_child", "right_child"] 
    feature_index: int
    threshold: float
    left_child: Node
    right_child: Node

    @override
    def predict(self, features: NDArray):
        if features[self.feature_index] <= self.threshold:
            return self.left_child.predict(features)
        else:
            return self.right_child.predict(features)
