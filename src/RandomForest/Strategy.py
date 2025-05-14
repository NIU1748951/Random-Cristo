from abc import ABC, abstractmethod
from collections import Counter
import math
from typing import override
import numpy as np
from numpy.typing import NDArray
from utils.LoggerConfig import get_logger

logger = get_logger(__name__)

class ImpurityStrategy(ABC):
    """
    Abstract class that defines the common interface for all impurity strategies.

    Every subclass must implement the method 'execute', which computes the impurity
    measure of a given array of labels.
    """
    def proportionate(self, labels: NDArray):
        """
        Calculate class proportions from label counts.
        
        Args:
            labels: Array of class labels
            
        Returns:
            List of class proportions
        """

        if len(labels) == 0:
            return 0
        counter = Counter(labels)
        proportions = [count/len(labels) for count in counter.values()]
        return proportions

    @abstractmethod
    def execute(self, labels: NDArray) -> float:
        """
        Calculate impurity measure for given labels.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Calculated impurity value
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass

class ImpurityStrategyGini(ImpurityStrategy):
    """Gini impurity calculation strategy using Gini formula."""
    @override
    def execute(self, labels: NDArray) -> float:
        """
        Compute Gini impurity for class label distribution.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Gini impurity value between 0 (pure) and 1 (max impurity)
        """

        proportions = self.proportionate(labels)
        if proportions == 0:
            return 0
        # G(D) = 1 - Σ p_c²
        return 1 - sum(p**2 for p in proportions)

    def __str__(self) -> str:
        return "GINI"

class ImpurityStrategyEntropy(ImpurityStrategy):
    """Entropy impurity calculation strategy using entropy formula."""
    @override
    def execute(self, labels: NDArray) -> float:
        """
        Compute information entropy for class label distribution.
        
        Args:
            labels: Array of class labels
            
        Returns:
            Entropy value in bits (base 2)
        """
        proportions = self.proportionate(labels)
        if proportions == 0:
            return 0
        #H(D) = -∑ p_c log p_c
        return -sum(p * math.log(p, 2) for p in proportions)

    def __str__(self) -> str:
        return "ENTROPY"

class ImpurityStrategySSE(ImpurityStrategy):
    """Entropy impurity calculation strategy using Sum of Squared Errors (SSE) formula."""
    @override
    def execute(self, labels: NDArray) -> float:
        if len(labels) == 0:
            return 0

        mu = labels.mean()
        return np.mean((labels - mu) ** 2)

    def __str__(self) -> str:
        return "SSE"

class ImpurityStrategyUnknown(ImpurityStrategy):
    """
    Placeholder strategy for unknown/unimplemented impurity calculations.
    
    WARNING:
        Should NEVER be instantiated directly. Used as fallback for debugging.
    """
    def __init__(self, name):
        logger.warning(f"Unknown Impurity algorithm '{name}' has been created!")
        self.name = name

    @override
    def execute(self, labels: NDArray):
        logger.error(f"Impurity algorithm '{self.name}' is not implemented!")
        raise NotImplementedError("Unkown algorithm")
