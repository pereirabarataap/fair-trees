# Trimmed for fair-trees: only accuracy_score and r2_score
from fair_trees.metrics._classification import accuracy_score
from fair_trees.metrics._regression import r2_score

__all__ = ["accuracy_score", "r2_score"]
