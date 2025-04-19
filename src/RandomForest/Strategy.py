from abc import ABC, abstractmethod
from collections import Counter
import math
from typing import override
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
        if len(labels) == 0:
            return 0
        counter = Counter(labels)
        proportions = [count/len(labels) for count in counter.values()]
        return proportions

    @abstractmethod
    def execute(self, labels: NDArray) -> float:
        pass

class ImpurityStrategyGini(ImpurityStrategy):
    @override
    def execute(self, labels: NDArray) -> float:
        proportions = self.proportionate(labels)
        if proportions == 0:
            return 0
        # G(D) = 1 - Σ p_c²
        return 1 - sum(p**2 for p in proportions)

    def __str__(self) -> str:
        return "GINI"

class ImpurityStrategyEntropy(ImpurityStrategy):
    @override
    def execute(self, labels: NDArray) -> float:
        proportions = self.proportionate(labels)
        if proportions == 0:
            return 0
        #H(D) = -∑ p_c log p_c
        return -sum(p * math.log(p, 2) for p in proportions)

    def __str__(self) -> str:
        return "ENTROPY"

class ImpurityStrategyUnknown(ImpurityStrategy):
    """
    This class represents an unimplemented impurity algorithm and should never be either created
    or called manually. Use just for debugging purposes.
    """
    def __init__(self, name):
        logger.warning(f"Unknown Impurity algorithm '{name}' has been created!")
        self.name = name

    @override
    def execute(self, _):
        logger.error(f"Impurity algorithm '{self.name}' is not implemented!")
        raise NotImplementedError("Unkown algorithm")
