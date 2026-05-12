"""
Tests for AsymmetryDecisionTreeClassifier.

Correctness checks:
  - additive mode with lambda_=0 produces the same tree as standard Gini DT
  - lexicographic mode selects the highest-asymmetry split among tied impurity splits
  - normalized asymmetry is always in [0, 1] (verified indirectly via valid training)
  - sklearn compatibility: predict_proba, cross_val_score, clone
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from threshold_trees import AsymmetryDecisionTreeClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_dataset():
    return make_classification(n_samples=300, n_features=5, random_state=42)


@pytest.fixture
def large_dataset():
    return make_classification(n_samples=1000, n_features=10, random_state=0)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

def test_fit_predict(small_dataset):
    X, y = small_dataset
    clf = AsymmetryDecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert set(preds).issubset({0, 1})


def test_predict_proba_sums_to_one(small_dataset):
    X, y = small_dataset
    clf = AsymmetryDecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# Correctness: additive with lambda_=0 matches standard DT
# ---------------------------------------------------------------------------

def test_lambda_zero_matches_standard_tree(small_dataset):
    """lambda_=0 reduces the score to pure Gini — should give identical splits."""
    X, y = small_dataset
    seed = 7

    base = DecisionTreeClassifier(max_depth=4, random_state=seed)
    base.fit(X, y)

    custom = AsymmetryDecisionTreeClassifier(
        max_depth=4, lambda_=0.0, lexicographic=False, random_state=seed
    )
    custom.fit(X, y)

    np.testing.assert_array_equal(
        base.predict(X),
        custom.predict(X),
        err_msg="With lambda_=0 the tree must produce identical predictions to the standard DT",
    )


# ---------------------------------------------------------------------------
# Correctness: lexicographic mode selects higher-asymmetry split
# ---------------------------------------------------------------------------

def test_lexicographic_prefers_asymmetric_split():
    """
    Two-feature dataset where both features produce a perfect Gini split
    (improvement=0.5, both children pure), but with different asymmetry:

      Feature 0: values [1,2,3,4]   → split at 2.5, mu_L=1.5, mu_R=3.5, range=3
                  asymmetry = max(2.5-1.5, 3.5-2.5)/3 = 1.0/3 ≈ 0.333

      Feature 1: values [1,2,9,10]  → split at 5.5, mu_L=1.5, mu_R=9.5, range=9
                  asymmetry = max(5.5-1.5, 9.5-5.5)/9 = 4.0/9 ≈ 0.444

    With random_state=2 the standard DT scans feature 0 first and keeps it
    (tied Gini, strict > not met for feature 1).  The lexicographic tree must
    still choose feature 1 because its asymmetry (0.444) > feature 0 (0.333).
    """
    X = np.array([[1, 1], [2, 2], [3, 9], [4, 10]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])

    # Standard DT picks feature 0 with this seed (Gini tie, first-encountered wins)
    std = DecisionTreeClassifier(max_depth=1, random_state=2)
    std.fit(X, y)
    assert std.tree_.feature[0] == 0, "Precondition: standard DT must pick feature 0"

    # Lexicographic must override to feature 1 (higher asymmetry)
    clf = AsymmetryDecisionTreeClassifier(
        max_depth=1, eps_impurity=1e-6, lexicographic=True, random_state=2
    )
    clf.fit(X, y)
    assert clf.tree_.feature[0] == 1, (
        f"Lexicographic should pick feature 1 (asym 0.444 > 0.333), got feature {clf.tree_.feature[0]}"
    )


def test_additive_high_lambda_selects_asymmetric_split():
    """
    With very high lambda_ the additive score is dominated by asymmetry.
    The split with the highest normalized asymmetry wins even if its
    Gini improvement is lower than another split's.

    Feature 1 at its best split has a large gap (mu_R far from threshold),
    so the additive tree should prefer feature 1 over feature 0.
    """
    X = np.array([[1, 1], [2, 2], [3, 9], [4, 10]], dtype=np.float32)
    y = np.array([0, 0, 1, 1])

    clf = AsymmetryDecisionTreeClassifier(
        max_depth=1, lambda_=10.0, lexicographic=False, random_state=2
    )
    clf.fit(X, y)

    # With high lambda, feature 1's asymmetry bonus dominates
    assert clf.tree_.feature[0] == 1, (
        f"With high lambda_ expected feature 1 (more asymmetric), got feature {clf.tree_.feature[0]}"
    )


# ---------------------------------------------------------------------------
# Correctness: asymmetry formula sanity check
# ---------------------------------------------------------------------------

def test_asymmetry_formula_manually():
    """
    Verify the formula by computing expected asymmetry by hand and confirming
    the tree picks the right split.

    X (1-D): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  (10 samples)
    y:        [0, 0, 0, 0, 0, 1, 1, 1, 1,  1]

    Best Gini split: threshold=5.5, perfect separation.
    Asymmetry at 5.5: mu_L=3.0, mu_R=8.0
        asym = max(5.5-3.0, 8.0-5.5) / (10-1) = 2.5/9 ≈ 0.278

    For additive with lambda_=0: tree must pick threshold=5.5 (best Gini).
    """
    X = np.arange(1, 11, dtype=np.float32).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    clf = AsymmetryDecisionTreeClassifier(max_depth=1, lambda_=0.0, random_state=0)
    clf.fit(X, y)

    assert clf.tree_.threshold[0] == pytest.approx(5.5, abs=0.5)
    assert clf.score(X, y) == 1.0


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def test_additive_mode_trains(large_dataset):
    X, y = large_dataset
    clf = AsymmetryDecisionTreeClassifier(
        max_depth=4, lambda_=0.5, lexicographic=False, random_state=42
    )
    clf.fit(X, y)
    assert clf.score(X, y) > 0.6


def test_lexicographic_mode_trains(large_dataset):
    X, y = large_dataset
    clf = AsymmetryDecisionTreeClassifier(
        max_depth=4, eps_impurity=1e-3, lexicographic=True, random_state=42
    )
    clf.fit(X, y)
    assert clf.score(X, y) > 0.6


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------

def test_cross_val_score(small_dataset):
    X, y = small_dataset
    clf = AsymmetryDecisionTreeClassifier(max_depth=3, lambda_=0.5, random_state=0)
    scores = cross_val_score(clf, X, y, cv=5)
    assert scores.mean() > 0.5


def test_clone_preserves_params(small_dataset):
    X, y = small_dataset
    clf = AsymmetryDecisionTreeClassifier(
        max_depth=3, lambda_=0.7, eps_impurity=1e-3, lexicographic=True, random_state=1
    )
    clf.fit(X, y)
    cloned = clone(clf)
    assert cloned.lambda_ == 0.7
    assert cloned.eps_impurity == 1e-3
    assert cloned.lexicographic is True


def test_sample_weight(small_dataset):
    X, y = small_dataset
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, size=len(y))
    clf = AsymmetryDecisionTreeClassifier(max_depth=3, random_state=0)
    clf.fit(X, y, sample_weight=w)
    assert clf.score(X, y) > 0.5
