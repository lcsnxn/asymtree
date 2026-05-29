import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._criterion import Gini
from sklearn.tree._tree import Tree, DepthFirstTreeBuilder
from sklearn.utils import check_random_state

try:
    from ._splitter import MarginBestSplitter          # compiled from _splitter.pyx
except ImportError:
    from .margin_splitter import MarginBestSplitter    # pre-built fallback (dev only)


class AsymmetryDecisionTreeClassifier(DecisionTreeClassifier):
    """Decision tree that rewards splits where one side's centroid is far
    from the splitting threshold (centroid asymmetry).

    For a candidate split on feature k at threshold t, with left samples L
    and right samples R, the asymmetry score is:

        asymmetry(k, t) = max(t - mean(L_k), mean(R_k) - t) / range(X_k)

    This is always in [0, 1] and is large when one centroid is pulled far
    from the boundary.

    Parameters
    ----------
    lambda_ : float, default=1.0
        Weight of the asymmetry term (additive mode only).
    eps_impurity : float, default=1e-4
        Tolerance band for the lexicographic tiebreak. Splits within
        `eps_impurity` of the best Gini improvement compete on asymmetry.
    lexicographic : bool, default=False
        If True, use lexicographic mode (purity first, asymmetry as
        tiebreaker among splits within eps_impurity of the best).
        If False, use additive mode:
            score = impurity_improvement + lambda_ * asymmetry
    All other parameters are identical to DecisionTreeClassifier.
    """

    def __init__(
        self,
        *,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        min_impurity_decrease=0.0,
        lambda_=1.0,
        eps_impurity=1e-4,
        lexicographic=False,
        **kwargs,
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            **kwargs,
        )
        self.lambda_ = lambda_
        self.eps_impurity = eps_impurity
        self.lexicographic = lexicographic

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        self._rebuild_with_custom_splitter(X, y, sample_weight)
        return self

    def _rebuild_with_custom_splitter(self, X, y, sample_weight):
        rng = check_random_state(self.random_state)

        criterion = Gini(
            self.n_outputs_,
            np.array([self.n_classes_], dtype=np.intp),
        )

        splitter = MarginBestSplitter(
            criterion,
            max_features=self.max_features_,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_leaf=0.0,
            random_state=rng,
            monotonic_cst=None,
        )
        # Set asymmetry params after construction (Cython __cinit__ rejects extras).
        splitter.lambda_ = self.lambda_
        splitter.eps_impurity = self.eps_impurity
        splitter.lexicographic = self.lexicographic

        tree = Tree(
            self.n_features_in_,
            np.array([self.n_classes_], dtype=np.intp),
            self.n_outputs_,
        )

        builder = DepthFirstTreeBuilder(
            splitter,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_leaf=0.0,
            max_depth=self.max_depth if self.max_depth else 2**31 - 1,
            min_impurity_decrease=self.min_impurity_decrease,
        )

        X_f32 = np.asarray(X, dtype=np.float32, order="C")
        y_enc = (
            np.searchsorted(self.classes_, y)
            .reshape(-1, 1)
            .astype(np.float64)
        )
        builder.build(tree, X_f32, np.ascontiguousarray(y_enc), sample_weight)
        self.tree_ = tree
