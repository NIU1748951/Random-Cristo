from dataclasses import dataclass
from typing import override
from abc import ABC, abstractmethod
from numpy.typing import NDArray

#NOTE: Our Docstrings follow the PEP257 standards 

class Node(ABC):
    """Abstract base class representing a node in a decision tree.
    
    Provides the interface for making predictions based on input features.
    """

    @abstractmethod
    def predict(self, features: NDArray):
        """Predict class label for a single sample's features.
        
        Args:
            features: Input feature array for a single sample
            
        Returns:
            int: Predicted class label
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

@dataclass
class Leaf(Node):
    """Terminal node containing final class prediction.
    
    Attributes:
        label: Constant class label this leaf node returns
    """
    __slots__ = ["label"] 
    label: int

    @override
    def predict(self, _: NDArray):
        """Return the leaf's stored class label.
        
        Args:
            _: Unused features parameter (maintains interface compatibility)
            
        Returns:
            int: Pre-determined class label
        """
        return self.label

@dataclass
class Parent(Node):
    """Internal decision node containing split criteria and child nodes.
    
    Attributes:
        feature_index: Index of feature used for splitting
        threshold: Value threshold for splitting
        left_child: Node handling samples <= threshold
        right_child: Node handling samples > threshold
    """
    __slots__ = ["feature_index", "threshold", "left_child", "right_child"] 
    feature_index: int
    threshold: float
    left_child: Node
    right_child: Node

    @override
    def predict(self, features: NDArray):
        """Route sample to appropriate child node based on split criteria.
        
        Args:
            features: Input feature array for a single sample
            
        Returns:
            int: Predicted class label from child node
            
        Raises:
            IndexError: If feature_index exceeds features dimensions
        """
        if features[self.feature_index] <= self.threshold:
            return self.left_child.predict(features)
        else:
            return self.right_child.predict(features)
