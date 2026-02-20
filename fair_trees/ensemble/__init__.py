# Fair ensemble: AUC-based random forest classifier only
from fair_trees.ensemble._base import BaseEnsemble
from fair_trees.ensemble._forest import FairRandomForestClassifier

__all__ = [
    "BaseEnsemble",
    "FairRandomForestClassifier",
]
