from abc import ABC, abstractmethod
from collections import defaultdict
from .DecisionTree import Leaf
from utils.LoggerConfig import get_logger

logger = get_logger(__name__)

class TreeVisitor(ABC):
    """Base visitor for traversing decision tree nodes."""
    @abstractmethod
    def visit(self, node):
        """Perform an operation on the given node.

        Args:
            node: A decision tree node (Parent or Leaf) to visit.
        """
        pass

class FeatureImportanceVisitor(TreeVisitor):
    """Visitor that accumulates feature split counts for importance."""
    def __init__(self):
        self.counts = defaultdict(int)

    def visit(self, node):
        """Visit a node and increment the split count for its feature, if applicable.

        Args:
            node: A Parent or Leaf node from the decision tree.
        """
        if hasattr(node, 'feature_index') and node.feature_index is not None:
            self.counts[node.feature_index] += 1

    def get_importances(self):
        """Compute normalized importance scores for each feature.

        Returns:
            dict: Mapping from feature index to normalized importance score (0-1).
        """
        total = sum(self.counts.values())
        if total == 0:
            return {}
        return {idx: count / total for idx, count in self.counts.items()}

class TreePrinterVisitor(TreeVisitor):
    """Visitor that logs the structure of a decision tree to file."""
    def __init__(self):
        self._indent = 0

    def visit(self, node):
        """Visit a node and log its details, then recurse into children.

        Logs either a leaf prediction or a parent split rule with proper indentation.

        Args:
            node: A Parent or Leaf node from the decision tree.
        """
        prefix = '  ' * self._indent
        #Leaf node
        if isinstance(node, Leaf):
            logger.info(f"{prefix}Leaf: predict={node.label}")
            return
        #Parent node
        logger.info(f"{prefix}Node: X[{node.feature_index}] <= {node.threshold}")
        self._indent += 1
        node.left_child.accept(self)
        node.right_child.accept(self)
        self._indent -= 1