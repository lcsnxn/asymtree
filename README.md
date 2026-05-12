# threshold-trees

A scikit-learn compatible decision tree classifier that rewards splits where one side's centroid is pulled far from the splitting boundary — **centroid asymmetry**.

## Motivation

Standard decision trees choose splits purely on impurity (Gini / entropy). A split that perfectly separates classes but places both centroids equidistant from the boundary is treated the same as one where nearly all of one class is packed tightly on one side. `threshold-trees` exposes this asymmetry as an explicit objective, so you can tune how aggressively the tree favors interpretable, one-sided splits.

## Mathematical background

For a candidate split on feature *k* at threshold *t*, with left samples  
**L** = {x : x_k ≤ t} and right samples **R** = {x : x_k > t}, define:

```
asymmetry(k, t) = max(t − μ_L, μ_R − t) / (x_k_max − x_k_min)
```

where μ_L and μ_R are the feature-k means on each side. The denominator
normalises by the feature range so scores are comparable across features and
always lie in **[0, 1]**.

### Two combination strategies

**Additive** — score the split as a weighted sum:

```
score(k, t) = ΔGini(k, t) + λ · asymmetry(k, t)
```

**Lexicographic** — among all splits within ε of the best Gini improvement,
pick the one with the highest asymmetry. Purity and asymmetry are fully
decoupled.

### Efficient implementation

μ_L and μ_R are maintained as running sums while the threshold scan moves
left to right, giving the same O(n) cost per feature as the standard split
search — no extra pass over the data.

## Installation

```bash
pip install threshold-trees
```

**Requirements**: Python ≥ 3.9, scikit-learn ≥ 1.4, numpy ≥ 1.21.  
A C compiler and Cython ≥ 3.0 are needed to build from source.

## Quick start

```python
from threshold_trees import AsymmetryDecisionTreeClassifier

# Additive mode: impurity + 0.5 × asymmetry
clf = AsymmetryDecisionTreeClassifier(
    max_depth=4,
    lambda_=0.5,
    lexicographic=False,
    random_state=42,
)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

# Lexicographic mode: purity first, asymmetry breaks ties
clf_lexico = AsymmetryDecisionTreeClassifier(
    max_depth=4,
    eps_impurity=1e-3,
    lexicographic=True,
    random_state=42,
)
clf_lexico.fit(X_train, y_train)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_` | float | 1.0 | Weight of the asymmetry term (additive mode only). |
| `eps_impurity` | float | 1e-4 | Tolerance band for lexicographic tiebreaking. |
| `lexicographic` | bool | False | If True, use lexicographic mode. |
| `max_depth` | int \| None | None | Maximum tree depth. |
| `min_samples_split` | int | 2 | Minimum samples to split a node. |
| `min_samples_leaf` | int | 1 | Minimum samples in a leaf. |
| `max_features` | int \| float \| str \| None | None | Number of features to consider per split. |
| `random_state` | int \| None | None | Random seed. |

All other `DecisionTreeClassifier` parameters are forwarded unchanged.

## Compatibility

`AsymmetryDecisionTreeClassifier` is a drop-in replacement for
`sklearn.tree.DecisionTreeClassifier` and works with all sklearn utilities:
cross-validation, pipelines, `clone`, `GridSearchCV`, `plot_tree`, etc.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [3, 4, 5], "lambda_": [0.1, 0.5, 1.0]}
gs = GridSearchCV(AsymmetryDecisionTreeClassifier(), param_grid, cv=5)
gs.fit(X_train, y_train)
```

## Running the tests

```bash
pip install pytest
pytest tests/
```

## License

MIT
