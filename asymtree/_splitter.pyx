# cython: boundscheck=False, wraparound=False, cdivision=True
"""
MarginBestSplitter — extends Splitter with a centroid-asymmetry bonus.

Asymmetry score (normalized to [0, 1]):
    asymmetry(k, t) = max(t - mu_L, mu_R - t) / (x_max - x_min)

Additive objective:
    score(k, t) = impurity_improvement(k, t) + lambda_ * asymmetry(k, t)

Lexicographic objective:
    Among splits within eps_impurity of the best impurity improvement,
    pick the one with the highest asymmetry.

Incremental centroid updates (O(1) per threshold step):
    mu_L and mu_R are maintained as running sums divided by sample counts.
"""

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy

from sklearn.tree._splitter cimport Splitter, SplitRecord
from sklearn.tree._criterion cimport Criterion
from sklearn.tree._tree cimport ParentInfo
from sklearn.tree._partitioner cimport (
    DensePartitioner,
    FEATURE_THRESHOLD,
    shift_missing_values_to_left_if_required,
)
from sklearn.tree._utils cimport rand_int
from sklearn.utils._typedefs cimport (
    float32_t, float64_t, intp_t, uint8_t, uint32_t, int8_t,
)

cdef float64_t INFINITY = np.inf

DTYPE = np.float32


cdef inline void _init_split(SplitRecord* s, intp_t end_pos) noexcept nogil:
    s.impurity_left      = INFINITY
    s.impurity_right     = INFINITY
    s.pos                = end_pos
    s.feature            = 0
    s.threshold          = 0.0
    s.improvement        = -INFINITY
    s.missing_go_to_left = False
    s.n_missing          = 0


cdef class MarginBestSplitter(Splitter):

    cdef DensePartitioner partitioner
    cdef public float64_t lambda_
    cdef public float64_t eps_impurity
    cdef public bint      lexicographic

    def __cinit__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst=None,
    ):
        pass  # Splitter.__cinit__ handles base initialization

    def __init__(
        self,
        Criterion criterion,
        intp_t max_features,
        intp_t min_samples_leaf,
        float64_t min_weight_leaf,
        object random_state,
        const int8_t[:] monotonic_cst=None,
        float64_t lambda_=1.0,
        float64_t eps_impurity=1e-4,
        bint lexicographic=False,
    ):
        self.lambda_       = lambda_
        self.eps_impurity  = eps_impurity
        self.lexicographic = lexicographic

    cdef int init(
        self,
        object X,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        const uint8_t[::1] missing_values_in_feature_mask,
    ) except -1:
        Splitter.init(self, X, y, sample_weight, missing_values_in_feature_mask)
        self.partitioner = DensePartitioner(
            X, self.samples, self.feature_values, missing_values_in_feature_mask
        )
        return 0

    cdef int node_split(
        self,
        ParentInfo* parent_record,
        SplitRecord* split,
    ) except -1 nogil:

        cdef intp_t start = self.start
        cdef intp_t end   = self.end
        cdef intp_t end_non_missing
        cdef intp_t n_missing = 0
        cdef bint   has_missing = False

        cdef intp_t[::1]    samples           = self.samples
        cdef intp_t[::1]    features          = self.features
        cdef intp_t[::1]    constant_features = self.constant_features
        cdef intp_t         n_features        = self.n_features
        cdef float32_t[::1] feature_values    = self.feature_values
        cdef intp_t         max_features      = self.max_features
        cdef intp_t         min_samples_leaf  = self.min_samples_leaf
        cdef float64_t      min_weight_leaf   = self.min_weight_leaf
        cdef uint32_t*      random_state      = &self.rand_r_state
        cdef float64_t      impurity          = parent_record.impurity

        # Fisher-Yates feature sampling counters
        cdef intp_t f_i = n_features
        cdef intp_t f_j
        cdef intp_t p, p_prev, p_before
        cdef intp_t n_visited_features = 0
        cdef intp_t n_found_constants  = 0
        cdef intp_t n_drawn_constants  = 0
        cdef intp_t n_known_constants  = parent_record.n_constant_features
        cdef intp_t n_total_constants  = n_known_constants

        # Best-split trackers
        cdef SplitRecord best_split, current_split
        cdef float64_t   best_score       = -INFINITY
        cdef float64_t   best_improvement = -INFINITY
        cdef float64_t   best_asymmetry   = -INFINITY

        # Per-split quantities
        cdef float64_t impurity_left, impurity_right
        cdef float64_t improvement, asymmetry, score
        cdef float64_t threshold
        cdef float64_t feature_range
        cdef float64_t sum_left, sum_right, mu_left, mu_right
        cdef intp_t    n_left_cnt, n_right_cnt, k
        cdef intp_t    n_left, n_right

        _init_split(&best_split, end)
        self.partitioner.init_node_split(start, end)

        while (f_i > n_total_constants and
               (n_visited_features < max_features or
                n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            f_j = rand_int(n_drawn_constants, f_i - n_found_constants, random_state)

            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = (
                    features[f_j], features[n_drawn_constants]
                )
                n_drawn_constants += 1
                continue

            f_j += n_found_constants
            current_split.feature = features[f_j]
            self.partitioner.sort_samples_and_feature_values(current_split.feature)
            n_missing       = self.partitioner.n_missing
            end_non_missing = end - n_missing

            if (end_non_missing == start or
                (feature_values[end_non_missing - 1] <=
                 feature_values[start] + FEATURE_THRESHOLD and n_missing == 0)):
                features[f_j], features[n_total_constants] = (
                    features[n_total_constants], features[f_j]
                )
                n_found_constants += 1
                n_total_constants += 1
                continue

            f_i -= 1
            features[f_i], features[f_j] = features[f_j], features[f_i]
            has_missing = n_missing != 0
            self.criterion.init_missing(n_missing)

            # Feature range for asymmetry normalisation
            feature_range = (<float64_t>feature_values[end_non_missing - 1] -
                             <float64_t>feature_values[start])
            if feature_range < 1e-10:
                feature_range = 1.0

            # Running centroid sums — initially all non-missing samples are on the right
            sum_right   = 0.0
            for k in range(start, end_non_missing):
                sum_right += feature_values[k]
            sum_left    = 0.0
            n_left_cnt  = 0
            n_right_cnt = end_non_missing - start

            self.criterion.missing_go_to_left = False
            self.criterion.reset()

            p      = start
            p_prev = start

            while p < end_non_missing:
                p_before = p
                self.partitioner.next_p(&p_prev, &p)

                if p >= end_non_missing:
                    break

                # Move samples [p_before, p) from right to left
                for k in range(p_before, p):
                    sum_left    += feature_values[k]
                    sum_right   -= feature_values[k]
                    n_left_cnt  += 1
                    n_right_cnt -= 1

                n_left  = n_left_cnt
                n_right = n_right_cnt + n_missing

                if n_left < min_samples_leaf or n_right < min_samples_leaf:
                    continue

                current_split.pos = p
                self.criterion.update(current_split.pos)

                if (self.criterion.weighted_n_left  < min_weight_leaf or
                        self.criterion.weighted_n_right < min_weight_leaf):
                    continue

                self.criterion.children_impurity(&impurity_left, &impurity_right)
                improvement = self.criterion.impurity_improvement(
                    impurity, impurity_left, impurity_right
                )

                # Midpoint threshold between adjacent distinct values
                threshold = (<float64_t>feature_values[p_prev] / 2.0 +
                             <float64_t>feature_values[p]      / 2.0)
                if (threshold == feature_values[p] or
                        threshold ==  INFINITY or
                        threshold == -INFINITY):
                    threshold = feature_values[p_prev]

                # Normalized centroid asymmetry
                if n_left_cnt > 0 and n_right_cnt > 0:
                    mu_left   = sum_left  / n_left_cnt
                    mu_right  = sum_right / n_right_cnt
                    asymmetry = max(threshold - mu_left,
                                   mu_right  - threshold) / feature_range
                else:
                    asymmetry = 0.0

                # ── ADDITIVE mode ──────────────────────────────────────────
                if not self.lexicographic:
                    score = improvement + self.lambda_ * asymmetry
                    if score > best_score:
                        best_score                    = score
                        best_split.feature            = current_split.feature
                        best_split.pos                = p
                        best_split.threshold          = threshold
                        best_split.improvement        = improvement
                        best_split.impurity_left      = impurity_left
                        best_split.impurity_right     = impurity_right
                        best_split.n_missing          = n_missing
                        best_split.missing_go_to_left = (
                            n_left_cnt > n_right_cnt if n_missing == 0 else False
                        )

                # ── LEXICOGRAPHIC mode ─────────────────────────────────────
                else:
                    if improvement > best_improvement + self.eps_impurity:
                        best_improvement              = improvement
                        best_asymmetry                = asymmetry
                        best_split.feature            = current_split.feature
                        best_split.pos                = p
                        best_split.threshold          = threshold
                        best_split.improvement        = improvement
                        best_split.impurity_left      = impurity_left
                        best_split.impurity_right     = impurity_right
                        best_split.n_missing          = n_missing
                        best_split.missing_go_to_left = (
                            n_left_cnt > n_right_cnt if n_missing == 0 else False
                        )

                    elif (improvement >= best_improvement - self.eps_impurity and
                          asymmetry > best_asymmetry):
                        best_asymmetry                = asymmetry
                        best_split.feature            = current_split.feature
                        best_split.pos                = p
                        best_split.threshold          = threshold
                        best_split.improvement        = improvement
                        best_split.impurity_left      = impurity_left
                        best_split.impurity_right     = impurity_right
                        best_split.n_missing          = n_missing
                        best_split.missing_go_to_left = (
                            n_left_cnt > n_right_cnt if n_missing == 0 else False
                        )

        # ── Finalise best split ────────────────────────────────────────────
        if best_split.pos < end:
            self.partitioner.partition_samples_final(
                best_split.pos,
                best_split.threshold,
                best_split.feature,
                best_split.n_missing,
            )
            self.criterion.init_missing(best_split.n_missing)
            self.criterion.missing_go_to_left = best_split.missing_go_to_left
            self.criterion.reset()
            self.criterion.update(best_split.pos)
            self.criterion.children_impurity(
                &best_split.impurity_left, &best_split.impurity_right
            )
            best_split.improvement = self.criterion.impurity_improvement(
                impurity, best_split.impurity_left, best_split.impurity_right
            )
            shift_missing_values_to_left_if_required(&best_split, samples, end)

        memcpy(
            &features[0], &constant_features[0],
            sizeof(intp_t) * n_known_constants,
        )
        memcpy(
            &constant_features[n_known_constants],
            &features[n_known_constants],
            sizeof(intp_t) * n_found_constants,
        )

        parent_record.n_constant_features = n_total_constants
        split[0] = best_split
        return 0
