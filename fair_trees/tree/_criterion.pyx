# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from libc.string cimport memcpy
from libc.string cimport memset
from libc.math cimport fabs, INFINITY

import numpy as np
cimport numpy as cnp
cnp.import_array()

from scipy.special.cython_special cimport xlogy

from ._utils cimport log
from ._utils cimport WeightedMedianCalculator

# EPSILON is used in the Poisson criterion
cdef float64_t EPSILON = 10 * np.finfo('double').eps

cdef class Criterion:
    """Interface for impurity criteria.

    This object stores methods on how to calculate how good a split is using
    different metrics.
    """
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Placeholder for a method which will initialize the criterion.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            y is a buffer that can store values for n_outputs target variables
            stored as a Cython memoryview.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of the samples being considered
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to be used on this node
        end : intp_t
            The last sample used on this node

        """
        pass

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]

        Parameters
        ----------
        n_missing: intp_t
            Number of missing values for specific feature.
        """
        pass

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        This method must be implemented by the subclass.
        """
        pass

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        This method must be implemented by the subclass.
        """
        pass

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        This updates the collected statistics by moving sample_indices[pos:new_pos]
        from the right child to the left child. It must be implemented by
        the subclass.

        Parameters
        ----------
        new_pos : intp_t
            New starting index position of the sample_indices in the right child
        """
        pass

    cdef float64_t node_impurity(self) noexcept nogil:
        """Placeholder for calculating the impurity of the node.

        Placeholder for a method which will evaluate the impurity of
        the current node, i.e. the impurity of sample_indices[start:end]. This is the
        primary function of the criterion class. The smaller the impurity the
        better.
        """
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Placeholder for calculating the impurity of children.

        Placeholder for a method which evaluates the impurity in
        children nodes, i.e. the impurity of sample_indices[start:pos] + the impurity
        of sample_indices[pos:end].

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address where the impurity of the left child should be
            stored.
        impurity_right : float64_t pointer
            The memory address where the impurity of the right child should be
            stored
        """
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Placeholder for storing the node value.

        Placeholder for a method which will compute the node value
        of sample_indices[start:end] and save the value into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address where the node value should be stored.
        """
        pass

    cdef void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        pass

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints

        This method is implemented in ClassificationCriterion and RegressionCriterion.
        """
        pass

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t impurity_left
        cdef float64_t impurity_right
        self.children_impurity(&impurity_left, &impurity_right)

        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        """Compute the improvement in impurity.

        This method computes the improvement in impurity when a split occurs.
        The weighted impurity improvement equation is the following:

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where N is the total number of samples, N_t is the number of samples
        at the current node, N_t_L is the number of samples in the left child,
        and N_t_R is the number of samples in the right child,

        Parameters
        ----------
        impurity_parent : float64_t
            The initial impurity of the parent node before the split

        impurity_left : float64_t
            The impurity of the left child

        impurity_right : float64_t
            The impurity of the right child

        Return
        ------
        float64_t : improvement in impurity after the split occurs
        """
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent - (self.weighted_n_right /
                                    self.weighted_n_node_samples * impurity_right)
                                 - (self.weighted_n_left /
                                    self.weighted_n_node_samples * impurity_left)))

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        pass

    cdef inline bint _check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
        float64_t value_left,
        float64_t value_right,
    ) noexcept nogil:
        cdef:
            bint check_lower_bound = (
                (value_left >= lower_bound) &
                (value_right >= lower_bound)
            )
            bint check_upper_bound = (
                (value_left <= upper_bound) &
                (value_right <= upper_bound)
            )
            bint check_monotonic_cst = (
                (value_left - value_right) * monotonic_cst <= 0
            )
        return check_lower_bound & check_upper_bound & check_monotonic_cst

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""

cdef inline void _move_sums_classification(
    ClassificationCriterion criterion,
    float64_t[:, ::1] sum_1,
    float64_t[:, ::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef intp_t k, c, n_bytes
    if criterion.n_missing != 0 and put_missing_in_1:
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k] * sizeof(float64_t)
            memcpy(&sum_1[k, 0], &criterion.sum_missing[k, 0], n_bytes)

        for k in range(criterion.n_outputs):
            for c in range(criterion.n_classes[k]):
                sum_2[k, c] = criterion.sum_total[k, c] - criterion.sum_missing[k, c]

        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        # Assigning sum_2 = sum_total for all outputs.
        for k in range(criterion.n_outputs):
            n_bytes = criterion.n_classes[k] * sizeof(float64_t)
            memset(&sum_1[k, 0], 0, n_bytes)
            memcpy(&sum_2[k, 0], &criterion.sum_total[k, 0], n_bytes)

        weighted_n_1[0] = 0.0
        weighted_n_2[0] = criterion.weighted_n_node_samples


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""

    def __cinit__(self, intp_t n_outputs,
                  cnp.ndarray[intp_t, ndim=1] n_classes):
        """Initialize attributes for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets, the dimensionality of the prediction
        n_classes : numpy.ndarray, dtype=intp_t
            The number of unique classes in each target
        """
        self.start = 0
        self.pos = 0
        self.end = 0
        self.missing_go_to_left = 0

        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.n_classes = np.empty(n_outputs, dtype=np.intp)

        cdef intp_t k = 0
        cdef intp_t max_n_classes = 0

        # For each target, set the number of unique classes in that target,
        # and also compute the maximal stride of all targets
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > max_n_classes:
                max_n_classes = n_classes[k]

        self.max_n_classes = max_n_classes

        # Count labels for each output
        self.sum_total = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_left = np.zeros((n_outputs, max_n_classes), dtype=np.float64)
        self.sum_right = np.zeros((n_outputs, max_n_classes), dtype=np.float64)

    def __reduce__(self):
        return (type(self),
                (self.n_outputs, np.asarray(self.n_classes)), self.__getstate__())

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        y : ndarray, dtype=float64_t
            The target stored as a buffer for memory efficiency.
        sample_weight : ndarray, dtype=float64_t
            The weight of each sample stored as a Cython memoryview.
        weighted_n_samples : float64_t
            The total weight of all samples
        sample_indices : ndarray, dtype=intp_t
            A mask on the samples. Indices of the samples in X and y we want to use,
            where sample_indices[start:end] correspond to the samples in this node.
        start : intp_t
            The first sample to use in the mask
        end : intp_t
            The last sample to use in the mask
        """
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef intp_t c
        cdef float64_t w = 1.0

        for k in range(self.n_outputs):
            memset(&self.sum_total[k, 0], 0, self.n_classes[k] * sizeof(float64_t))

        for p in range(start, end):
            i = sample_indices[p]

            # w is originally set to be 1.0, meaning that if no sample weights
            # are given, the default weight of each sample is 1.0.
            if sample_weight is not None:
                w = sample_weight[i]

            # Count weighted class frequency for each target
            for k in range(self.n_outputs):
                c = <intp_t> self.y[i, k]
                self.sum_total[k, c] += w

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = np.zeros((self.n_outputs, self.max_n_classes), dtype=np.float64)

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, p, k, c
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.sum_missing[0, 0], 0, self.max_n_classes * self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                c = <intp_t> self.y[i, k]
                self.sum_missing[k, c] += w

            self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.start
        _move_sums_classification(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left,
        )
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.pos = self.end
        _move_sums_classification(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left
        )
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left child.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        new_pos : intp_t
            The new ending position for which to move sample_indices from the right
            child to the left child.
        """
        cdef intp_t pos = self.pos
        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing

        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef const float64_t[:] sample_weight = self.sample_weight

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef intp_t c
        cdef float64_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #   sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_po.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k, <intp_t> self.y[i, k]] += w

                self.weighted_n_left += w

        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k, <intp_t> self.y[i, k]] -= w

                self.weighted_n_left -= w

        # Update right part statistics
        self.weighted_n_right = self.weighted_n_node_samples - self.weighted_n_left
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                self.sum_right[k, c] = self.sum_total[k, c] - self.sum_left[k, c]

        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] and save it into dest.

        Parameters
        ----------
        dest : float64_t pointer
            The memory address which we will save the node value into.
        """
        cdef intp_t k, c

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                dest[c] = self.sum_total[k, c] / self.weighted_n_node_samples
            dest += self.max_n_classes

    cdef inline void clip_node_value(
        self, float64_t * dest, float64_t lower_bound, float64_t upper_bound
    ) noexcept nogil:
        """Clip the values in dest such that predicted probabilities stay between
        `lower_bound` and `upper_bound` when monotonic constraints are enforced.
        Note that monotonicity constraints are only supported for:
        - single-output trees and
        - binary classifications.
        """
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

        # Values for binary classification must sum to 1.
        dest[1] = 1 - dest[0]

    cdef inline float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Note that monotonicity constraints are only supported for:
        - single-output trees and
        - binary classifications.
        """
        return (
            (self.sum_left[0, 0] / (2 * self.weighted_n_left)) +
            (self.sum_right[0, 0] / (2 * self.weighted_n_right))
        )

    cdef inline bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current classification split"""
        cdef:
            float64_t value_left = self.sum_left[0][0] / self.weighted_n_left
            float64_t value_right = self.sum_right[0][0] / self.weighted_n_right

        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)


cdef class Entropy(ClassificationCriterion):
    r"""Cross Entropy impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1 / Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = -\sum_{k=0}^{K-1} count_k log(count_k)
    """

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the cross-entropy criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef float64_t entropy = 0.0
        cdef float64_t count_k
        cdef intp_t k
        cdef intp_t c

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_total[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)

        return entropy / self.n_outputs

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address to save the impurity of the left node
        impurity_right : float64_t pointer
            The memory address to save the impurity of the right node
        """
        cdef float64_t entropy_left = 0.0
        cdef float64_t entropy_right = 0.0
        cdef float64_t count_k
        cdef intp_t k
        cdef intp_t c

        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)

                count_k = self.sum_right[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)

        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs


cdef class Gini(ClassificationCriterion):
    r"""Gini Index impurity criterion.

    This handles cases where the target is a classification taking values
    0, 1, ... K-2, K-1. If node m represents a region Rm with Nm observations,
    then let

        count_k = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} count_k (1 - count_k)
              = 1 - \sum_{k=0}^{K-1} count_k ** 2
    """

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the Gini criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef float64_t gini = 0.0
        cdef float64_t sq_count
        cdef float64_t count_k
        cdef intp_t k
        cdef intp_t c

        for k in range(self.n_outputs):
            sq_count = 0.0

            for c in range(self.n_classes[k]):
                count_k = self.sum_total[k, c]
                sq_count += count_k * count_k

            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)

        return gini / self.n_outputs

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]) using the Gini index.

        Parameters
        ----------
        impurity_left : float64_t pointer
            The memory address to save the impurity of the left node to
        impurity_right : float64_t pointer
            The memory address to save the impurity of the right node to
        """
        cdef float64_t gini_left = 0.0
        cdef float64_t gini_right = 0.0
        cdef float64_t sq_count_left
        cdef float64_t sq_count_right
        cdef float64_t count_k
        cdef intp_t k
        cdef intp_t c

        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0

            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                sq_count_left += count_k * count_k

                count_k = self.sum_right[k, c]
                sq_count_right += count_k * count_k

            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)

            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)

        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef inline void _move_sums_regression(
    RegressionCriterion criterion,
    float64_t[::1] sum_1,
    float64_t[::1] sum_2,
    float64_t* weighted_n_1,
    float64_t* weighted_n_2,
    bint put_missing_in_1,
) noexcept nogil:
    """Distribute sum_total and sum_missing into sum_1 and sum_2.

    If there are missing values and:
    - put_missing_in_1 is True, then missing values to go sum_1. Specifically:
        sum_1 = sum_missing
        sum_2 = sum_total - sum_missing

    - put_missing_in_1 is False, then missing values go to sum_2. Specifically:
        sum_1 = 0
        sum_2 = sum_total
    """
    cdef:
        intp_t i
        intp_t n_bytes = criterion.n_outputs * sizeof(float64_t)
        bint has_missing = criterion.n_missing != 0

    if has_missing and put_missing_in_1:
        memcpy(&sum_1[0], &criterion.sum_missing[0], n_bytes)
        for i in range(criterion.n_outputs):
            sum_2[i] = criterion.sum_total[i] - criterion.sum_missing[i]
        weighted_n_1[0] = criterion.weighted_n_missing
        weighted_n_2[0] = criterion.weighted_n_node_samples - criterion.weighted_n_missing
    else:
        memset(&sum_1[0], 0, n_bytes)
        # Assigning sum_2 = sum_total for all outputs.
        memcpy(&sum_2[0], &criterion.sum_total[0], n_bytes)
        weighted_n_1[0] = 0.0
        weighted_n_2[0] = criterion.weighted_n_node_samples


cdef class RegressionCriterion(Criterion):
    r"""Abstract regression criterion.

    This handles cases where the target is a continuous value, and is
    evaluated by computing the variance of the target values left and right
    of the split point. The computation takes linear time with `n_samples`
    by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
    """

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.weighted_n_missing = 0.0

        self.sq_sum_total = 0.0

        self.sum_total = np.zeros(n_outputs, dtype=np.float64)
        self.sum_left = np.zeros(n_outputs, dtype=np.float64)
        self.sum_right = np.zeros(n_outputs, dtype=np.float64)

    def __reduce__(self):
        return (type(self), (self.n_outputs, self.n_samples), self.__getstate__())

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t y_ik
        cdef float64_t w_y_ik
        cdef float64_t w = 1.0
        self.sq_sum_total = 0.0
        memset(&self.sum_total[0], 0, self.n_outputs * sizeof(float64_t))

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_total[k] += w_y_ik
                self.sq_sum_total += w_y_ik * y_ik

            self.weighted_n_node_samples += w

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_sum_missing(self):
        """Init sum_missing to hold sums for missing values."""
        self.sum_missing = np.zeros(self.n_outputs, dtype=np.float64)

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize sum_missing if there are missing values.

        This method assumes that caller placed the missing samples in
        self.sample_indices[-n_missing:]
        """
        cdef intp_t i, p, k
        cdef float64_t y_ik
        cdef float64_t w_y_ik
        cdef float64_t w = 1.0

        self.n_missing = n_missing
        if n_missing == 0:
            return

        memset(&self.sum_missing[0], 0, self.n_outputs * sizeof(float64_t))

        self.weighted_n_missing = 0.0

        # The missing samples are assumed to be in self.sample_indices[-n_missing:]
        for p in range(self.end - n_missing, self.end):
            i = self.sample_indices[p]
            if self.sample_weight is not None:
                w = self.sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                w_y_ik = w * y_ik
                self.sum_missing[k] += w_y_ik

            self.weighted_n_missing += w

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start
        _move_sums_regression(
            self,
            self.sum_left,
            self.sum_right,
            &self.weighted_n_left,
            &self.weighted_n_right,
            self.missing_go_to_left
        )
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end."""
        self.pos = self.end
        _move_sums_regression(
            self,
            self.sum_right,
            self.sum_left,
            &self.weighted_n_right,
            &self.weighted_n_left,
            not self.missing_go_to_left
        )
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left."""
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t pos = self.pos

        # The missing samples are assumed to be in
        # self.sample_indices[-self.n_missing:] that is
        # self.sample_indices[end_non_missing:self.end].
        cdef intp_t end_non_missing = self.end - self.n_missing
        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0

        # Update statistics up to new_pos
        #
        # Given that
        #           sum_left[x] +  sum_right[x] = sum_total[x]
        # and that sum_total is known, we are going to update
        # sum_left from the direction that require the least amount
        # of computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end_non_missing - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k] += w * self.y[i, k]

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end_non_missing - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    self.sum_left[k] -= w * self.y[i, k]

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            self.sum_right[k] = self.sum_total[k] - self.sum_left[k]

        self.pos = new_pos
        return 0

    cdef float64_t node_impurity(self) noexcept nogil:
        pass

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        pass

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Compute the node value of sample_indices[start:end] into dest."""
        cdef intp_t k

        for k in range(self.n_outputs):
            dest[k] = self.sum_total[k] / self.weighted_n_node_samples

    cdef inline void clip_node_value(self, float64_t* dest, float64_t lower_bound, float64_t upper_bound) noexcept nogil:
        """Clip the value in dest between lower_bound and upper_bound for monotonic constraints."""
        if dest[0] < lower_bound:
            dest[0] = lower_bound
        elif dest[0] > upper_bound:
            dest[0] = upper_bound

    cdef float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        return (
            (self.sum_left[0] / (2 * self.weighted_n_left)) +
            (self.sum_right[0] / (2 * self.weighted_n_right))
        )

    cdef bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        cdef:
            float64_t value_left = self.sum_left[0] / self.weighted_n_left
            float64_t value_right = self.sum_right[0] / self.weighted_n_right

        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)

cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the MSE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef float64_t impurity
        cdef intp_t k

        impurity = self.sq_sum_total / self.weighted_n_node_samples
        for k in range(self.n_outputs):
            impurity -= (self.sum_total[k] / self.weighted_n_node_samples)**2.0

        return impurity / self.n_outputs

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The MSE proxy is derived from

            sum_{i left}(y_i - y_pred_L)^2 + sum_{i right}(y_i - y_pred_R)^2
            = sum(y_i^2) - n_L * mean_{i left}(y_i)^2 - n_R * mean_{i right}(y_i)^2

        Neglecting constant terms, this gives:

            - 1/n_L * sum_{i left}(y_i)^2 - 1/n_R * sum_{i right}(y_i)^2
        """
        cdef intp_t k
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0

        for k in range(self.n_outputs):
            proxy_impurity_left += self.sum_left[k] * self.sum_left[k]
            proxy_impurity_right += self.sum_right[k] * self.sum_right[k]

        return (proxy_impurity_left / self.weighted_n_left +
                proxy_impurity_right / self.weighted_n_right)

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t pos = self.pos
        cdef intp_t start = self.start

        cdef float64_t y_ik

        cdef float64_t sq_sum_left = 0.0
        cdef float64_t sq_sum_right

        cdef intp_t i
        cdef intp_t p
        cdef intp_t k
        cdef float64_t w = 1.0

        cdef intp_t end_non_missing

        for p in range(start, pos):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                y_ik = self.y[i, k]
                sq_sum_left += w * y_ik * y_ik

        if self.missing_go_to_left:
            # add up the impact of these missing values on the left child
            # statistics.
            # Note: this only impacts the square sum as the sum
            # is modified elsewhere.
            end_non_missing = self.end - self.n_missing

            for p in range(end_non_missing, self.end):
                i = sample_indices[p]
                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    y_ik = self.y[i, k]
                    sq_sum_left += w * y_ik * y_ik

        sq_sum_right = self.sq_sum_total - sq_sum_left

        impurity_left[0] = sq_sum_left / self.weighted_n_left
        impurity_right[0] = sq_sum_right / self.weighted_n_right

        for k in range(self.n_outputs):
            impurity_left[0] -= (self.sum_left[k] / self.weighted_n_left) ** 2.0
            impurity_right[0] -= (self.sum_right[k] / self.weighted_n_right) ** 2.0

        impurity_left[0] /= self.n_outputs
        impurity_right[0] /= self.n_outputs


cdef class MAE(RegressionCriterion):
    r"""Mean absolute error impurity criterion.

       MAE = (1 / n)*(\sum_i |y_i - f_i|), where y_i is the true
       value and f_i is the predicted value."""

    cdef cnp.ndarray left_child
    cdef cnp.ndarray right_child
    cdef void** left_child_ptr
    cdef void** right_child_ptr
    cdef float64_t[::1] node_medians

    def __cinit__(self, intp_t n_outputs, intp_t n_samples):
        """Initialize parameters for this criterion.

        Parameters
        ----------
        n_outputs : intp_t
            The number of targets to be predicted

        n_samples : intp_t
            The total number of samples to fit on
        """
        # Default values
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_samples = n_samples
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.node_medians = np.zeros(n_outputs, dtype=np.float64)

        self.left_child = np.empty(n_outputs, dtype='object')
        self.right_child = np.empty(n_outputs, dtype='object')
        # initialize WeightedMedianCalculators
        for k in range(n_outputs):
            self.left_child[k] = WeightedMedianCalculator(n_samples)
            self.right_child[k] = WeightedMedianCalculator(n_samples)

        self.left_child_ptr = <void**> cnp.PyArray_DATA(self.left_child)
        self.right_child_ptr = <void**> cnp.PyArray_DATA(self.right_child)

    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end,
    ) except -1 nogil:
        """Initialize the criterion.

        This initializes the criterion at node sample_indices[start:end] and children
        sample_indices[start:start] and sample_indices[start:end].
        """
        cdef intp_t i, p, k
        cdef float64_t w = 1.0

        # Initialize fields
        self.y = y
        self.sample_weight = sample_weight
        self.sample_indices = sample_indices
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.

        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        for k in range(self.n_outputs):
            (<WeightedMedianCalculator> left_child[k]).reset()
            (<WeightedMedianCalculator> right_child[k]).reset()

        for p in range(start, end):
            i = sample_indices[p]

            if sample_weight is not None:
                w = sample_weight[i]

            for k in range(self.n_outputs):
                # push method ends up calling safe_realloc, hence `except -1`
                # push all values to the right side,
                # since pos = start initially anyway
                (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

            self.weighted_n_node_samples += w
        # calculate the node medians
        for k in range(self.n_outputs):
            self.node_medians[k] = (<WeightedMedianCalculator> right_child[k]).get_median()

        # Reset to pos=start
        self.reset()
        return 0

    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Raise error if n_missing != 0."""
        if n_missing == 0:
            return
        with gil:
            raise ValueError("missing values is not supported for MAE.")

    cdef int reset(self) except -1 nogil:
        """Reset the criterion at pos=start.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef intp_t i, k
        cdef float64_t value
        cdef float64_t weight

        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        self.pos = self.start

        # reset the WeightedMedianCalculators, left should have no
        # elements and right should have all elements.

        for k in range(self.n_outputs):
            # if left has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> left_child[k]).size()):
                # remove everything from left and put it into right
                (<WeightedMedianCalculator> left_child[k]).pop(&value,
                                                               &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> right_child[k]).push(value,
                                                                 weight)
        return 0

    cdef int reverse_reset(self) except -1 nogil:
        """Reset the criterion at pos=end.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        self.weighted_n_right = 0.0
        self.weighted_n_left = self.weighted_n_node_samples
        self.pos = self.end

        cdef float64_t value
        cdef float64_t weight
        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        # reverse reset the WeightedMedianCalculators, right should have no
        # elements and left should have all elements.
        for k in range(self.n_outputs):
            # if right has no elements, it's already reset
            for i in range((<WeightedMedianCalculator> right_child[k]).size()):
                # remove everything from right and put it into left
                (<WeightedMedianCalculator> right_child[k]).pop(&value,
                                                                &weight)
                # push method ends up calling safe_realloc, hence `except -1`
                (<WeightedMedianCalculator> left_child[k]).push(value,
                                                                weight)
        return 0

    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Updated statistics by moving sample_indices[pos:new_pos] to the left.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        cdef intp_t pos = self.pos
        cdef intp_t end = self.end
        cdef intp_t i, p, k
        cdef float64_t w = 1.0

        # Update statistics up to new_pos
        #
        # We are going to update right_child and left_child
        # from the direction that require the least amount of
        # computations, i.e. from pos to new_pos or from end to new_pos.
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # remove y_ik and its weight w from right and add to left
                    (<WeightedMedianCalculator> right_child[k]).remove(self.y[i, k], w)
                    # push method ends up calling safe_realloc, hence except -1
                    (<WeightedMedianCalculator> left_child[k]).push(self.y[i, k], w)

                self.weighted_n_left += w
        else:
            self.reverse_reset()

            for p in range(end - 1, new_pos - 1, -1):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                for k in range(self.n_outputs):
                    # remove y_ik and its weight w from left and add to right
                    (<WeightedMedianCalculator> left_child[k]).remove(self.y[i, k], w)
                    (<WeightedMedianCalculator> right_child[k]).push(self.y[i, k], w)

                self.weighted_n_left -= w

        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        self.pos = new_pos
        return 0

    cdef void node_value(self, float64_t* dest) noexcept nogil:
        """Computes the node value of sample_indices[start:end] into dest."""
        cdef intp_t k
        for k in range(self.n_outputs):
            dest[k] = <float64_t> self.node_medians[k]

    cdef inline float64_t middle_value(self) noexcept nogil:
        """Compute the middle value of a split for monotonicity constraints as the simple average
        of the left and right children values.

        Monotonicity constraints are only supported for single-output trees we can safely assume
        n_outputs == 1.
        """
        return (
                (<WeightedMedianCalculator> self.left_child_ptr[0]).get_median() +
                (<WeightedMedianCalculator> self.right_child_ptr[0]).get_median()
        ) / 2

    cdef inline bint check_monotonicity(
        self,
        cnp.int8_t monotonic_cst,
        float64_t lower_bound,
        float64_t upper_bound,
    ) noexcept nogil:
        """Check monotonicity constraint is satisfied at the current regression split"""
        cdef:
            float64_t value_left = (<WeightedMedianCalculator> self.left_child_ptr[0]).get_median()
            float64_t value_right = (<WeightedMedianCalculator> self.right_child_ptr[0]).get_median()

        return self._check_monotonicity(monotonic_cst, lower_bound, upper_bound, value_left, value_right)

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the MAE criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices
        cdef intp_t i, p, k
        cdef float64_t w = 1.0
        cdef float64_t impurity = 0.0

        for k in range(self.n_outputs):
            for p in range(self.start, self.end):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                impurity += fabs(self.y[i, k] - self.node_medians[k]) * w

        return impurity / (self.weighted_n_node_samples * self.n_outputs)

    cdef void children_impurity(self, float64_t* p_impurity_left,
                                float64_t* p_impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity the right child (sample_indices[pos:end]).
        """
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef intp_t start = self.start
        cdef intp_t pos = self.pos
        cdef intp_t end = self.end

        cdef intp_t i, p, k
        cdef float64_t median
        cdef float64_t w = 1.0
        cdef float64_t impurity_left = 0.0
        cdef float64_t impurity_right = 0.0

        cdef void** left_child = self.left_child_ptr
        cdef void** right_child = self.right_child_ptr

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> left_child[k]).get_median()
            for p in range(start, pos):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                impurity_left += fabs(self.y[i, k] - median) * w
        p_impurity_left[0] = impurity_left / (self.weighted_n_left *
                                              self.n_outputs)

        for k in range(self.n_outputs):
            median = (<WeightedMedianCalculator> right_child[k]).get_median()
            for p in range(pos, end):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                impurity_right += fabs(self.y[i, k] - median) * w
        p_impurity_right[0] = impurity_right / (self.weighted_n_right *
                                                self.n_outputs)


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman.

    Uses the formula (35) in Friedman's original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.
        """
        cdef float64_t total_sum_left = 0.0
        cdef float64_t total_sum_right = 0.0

        cdef intp_t k
        cdef float64_t diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += self.sum_left[k]
            total_sum_right += self.sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right)

        return diff * diff / (self.weighted_n_left * self.weighted_n_right)

    cdef float64_t impurity_improvement(self, float64_t impurity_parent, float64_t
                                        impurity_left, float64_t impurity_right) noexcept nogil:
        # Note: none of the arguments are used here
        cdef float64_t total_sum_left = 0.0
        cdef float64_t total_sum_right = 0.0

        cdef intp_t k
        cdef float64_t diff = 0.0

        for k in range(self.n_outputs):
            total_sum_left += self.sum_left[k]
            total_sum_right += self.sum_right[k]

        diff = (self.weighted_n_right * total_sum_left -
                self.weighted_n_left * total_sum_right) / self.n_outputs

        return (diff * diff / (self.weighted_n_left * self.weighted_n_right *
                               self.weighted_n_node_samples))


cdef class Poisson(RegressionCriterion):
    """Half Poisson deviance as impurity criterion.

    Poisson deviance = 2/n * sum(y_true * log(y_true/y_pred) + y_pred - y_true)

    Note that the deviance is >= 0, and since we have `y_pred = mean(y_true)`
    at the leaves, one always has `sum(y_pred - y_true) = 0`. It remains the
    implemented impurity (factor 2 is skipped):
        1/n * sum(y_true * log(y_true/y_pred)
    """
    # FIXME in 1.0:
    # min_impurity_split with default = 0 forces us to use a non-negative
    # impurity like the Poisson deviance. Without this restriction, one could
    # throw away the 'constant' term sum(y_true * log(y_true)) and just use
    # Poisson loss = - 1/n * sum(y_true * log(y_pred))
    #              = - 1/n * sum(y_true * log(mean(y_true))
    #              = - mean(y_true) * log(mean(y_true))
    # With this trick (used in proxy_impurity_improvement()), as for MSE,
    # children_impurity would only need to go over left xor right split, not
    # both. This could be faster.

    cdef float64_t node_impurity(self) noexcept nogil:
        """Evaluate the impurity of the current node.

        Evaluate the Poisson criterion as impurity of the current node,
        i.e. the impurity of sample_indices[start:end]. The smaller the impurity the
        better.
        """
        return self.poisson_loss(self.start, self.end, self.sum_total,
                                 self.weighted_n_node_samples)

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute a proxy of the impurity reduction.

        This method is used to speed up the search for the best split.
        It is a proxy quantity such that the split that maximizes this value
        also maximizes the impurity improvement. It neglects all constant terms
        of the impurity decrease for a given split.

        The absolute impurity improvement is only computed by the
        impurity_improvement method once the best split has been found.

        The Poisson proxy is derived from:

              sum_{i left }(y_i * log(y_i / y_pred_L))
            + sum_{i right}(y_i * log(y_i / y_pred_R))
            = sum(y_i * log(y_i) - n_L * mean_{i left}(y_i) * log(mean_{i left}(y_i))
                                 - n_R * mean_{i right}(y_i) * log(mean_{i right}(y_i))

        Neglecting constant terms, this gives

            - sum{i left }(y_i) * log(mean{i left}(y_i))
            - sum{i right}(y_i) * log(mean{i right}(y_i))
        """
        cdef intp_t k
        cdef float64_t proxy_impurity_left = 0.0
        cdef float64_t proxy_impurity_right = 0.0
        cdef float64_t y_mean_left = 0.
        cdef float64_t y_mean_right = 0.

        for k in range(self.n_outputs):
            if (self.sum_left[k] <= EPSILON) or (self.sum_right[k] <= EPSILON):
                # Poisson loss does not allow non-positive predictions. We
                # therefore forbid splits that have child nodes with
                # sum(y_i) <= 0.
                # Since sum_right = sum_total - sum_left, it can lead to
                # floating point rounding error and will not give zero. Thus,
                # we relax the above comparison to sum(y_i) <= EPSILON.
                return -INFINITY
            else:
                y_mean_left = self.sum_left[k] / self.weighted_n_left
                y_mean_right = self.sum_right[k] / self.weighted_n_right
                proxy_impurity_left -= self.sum_left[k] * log(y_mean_left)
                proxy_impurity_right -= self.sum_right[k] * log(y_mean_right)

        return - proxy_impurity_left - proxy_impurity_right

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        """Evaluate the impurity in children nodes.

        i.e. the impurity of the left child (sample_indices[start:pos]) and the
        impurity of the right child (sample_indices[pos:end]) for Poisson.
        """
        cdef intp_t start = self.start
        cdef intp_t pos = self.pos
        cdef intp_t end = self.end

        impurity_left[0] = self.poisson_loss(start, pos, self.sum_left,
                                             self.weighted_n_left)

        impurity_right[0] = self.poisson_loss(pos, end, self.sum_right,
                                              self.weighted_n_right)

    cdef inline float64_t poisson_loss(
        self,
        intp_t start,
        intp_t end,
        const float64_t[::1] y_sum,
        float64_t weight_sum
    ) noexcept nogil:
        """Helper function to compute Poisson loss (~deviance) of a given node.
        """
        cdef const float64_t[:, ::1] y = self.y
        cdef const float64_t[:] sample_weight = self.sample_weight
        cdef const intp_t[:] sample_indices = self.sample_indices

        cdef float64_t y_mean = 0.
        cdef float64_t poisson_loss = 0.
        cdef float64_t w = 1.0
        cdef intp_t i, k, p
        cdef intp_t n_outputs = self.n_outputs

        for k in range(n_outputs):
            if y_sum[k] <= EPSILON:
                # y_sum could be computed from the subtraction
                # sum_right = sum_total - sum_left leading to a potential
                # floating point rounding error.
                # Thus, we relax the comparison y_sum <= 0 to
                # y_sum <= EPSILON.
                return INFINITY

            y_mean = y_sum[k] / weight_sum

            for p in range(start, end):
                i = sample_indices[p]

                if sample_weight is not None:
                    w = sample_weight[i]

                poisson_loss += w * xlogy(y[i, k], y[i, k] / y_mean)
        return poisson_loss / (weight_sum * n_outputs)


# =============================================================================
# Biased Classification Criteria with Z support
# =============================================================================

cdef class BiasedClassificationCriterion(ClassificationCriterion):
    """Base class for biased classification criteria with Z regularization.
    
    This criterion combines Y impurity (to minimize) with Z impurity (to maximize)
    using a linear combination controlled by theta:
    
    combined_impurity = (1 - theta) * impurity_y + theta * (-impurity_z)
    
    The goal is to minimize Y impurity while maximizing Z impurity (diversity).
    
    NOTE: Do not instantiate this class directly. Use create_biased_gini() or
    create_biased_entropy() factory functions instead.
    """
    
    def __reduce__(self):
        # For pickling support
        return (type(self),
                (self.n_outputs, np.asarray(self.n_classes),
                 np.asarray(self.z), np.asarray(self.n_z_classes),
                 self.theta, self.Z_agg_mode),
                self.__getstate__())
    
    cdef int init(
        self,
        const float64_t[:, ::1] y,
        const float64_t[:] sample_weight,
        float64_t weighted_n_samples,
        const intp_t[:] sample_indices,
        intp_t start,
        intp_t end
    ) except -1 nogil:
        """Initialize the criterion for a node."""
        # Declare all cdef variables at the beginning
        cdef intp_t i, p, k, c
        cdef float64_t w
        
        # First call parent init for Y
        ClassificationCriterion.init(
            self, y, sample_weight, weighted_n_samples,
            sample_indices, start, end
        )
        
        # Now initialize Z statistics
        if self.n_z_outputs > 0:
            # Reset Z sums
            for k in range(self.n_z_outputs):
                memset(&self.z_sum_total[k, 0], 0, self.n_z_classes[k] * sizeof(float64_t))
            
            # Accumulate Z statistics
            for p in range(start, end):
                i = sample_indices[p]
                
                if sample_weight is not None:
                    w = sample_weight[i]
                else:
                    w = 1.0
                
                # Count weighted class frequency for each Z column
                for k in range(self.n_z_outputs):
                    c = <intp_t> self.z[i, k]
                    self.z_sum_total[k, c] += w
        
        return 0
    
    cdef void init_sum_missing(self):
        """Initialize sum_missing for both Y and Z."""
        cdef intp_t k
        
        ClassificationCriterion.init_sum_missing(self)
        
        if self.n_z_outputs > 0:
            self.z_sum_missing = np.zeros((self.n_z_outputs, self.max_n_z_classes), dtype=np.float64)
    
    cdef void init_missing(self, intp_t n_missing) noexcept nogil:
        """Initialize missing value statistics for both Y and Z."""
        cdef intp_t i, p, k, c, start
        cdef float64_t w
        
        ClassificationCriterion.init_missing(self, n_missing)
        
        if self.n_z_outputs > 0 and n_missing > 0:
            start = self.end - n_missing
            
            # Reset Z missing sums
            for k in range(self.n_z_outputs):
                memset(&self.z_sum_missing[k, 0], 0, self.n_z_classes[k] * sizeof(float64_t))
            
            # Accumulate Z missing statistics
            for p in range(start, self.end):
                i = self.sample_indices[p]
                
                if self.sample_weight is not None:
                    w = self.sample_weight[i]
                else:
                    w = 1.0
                
                for k in range(self.n_z_outputs):
                    c = <intp_t> self.z[i, k]
                    self.z_sum_missing[k, c] += w
    
    cdef int reset(self) except -1 nogil:
        """Reset to pos=start for both Y and Z."""
        cdef intp_t k
        
        ClassificationCriterion.reset(self)
        
        if self.n_z_outputs > 0:
            # Initialize Z left/right sums
            if self.n_missing != 0:
                _move_sums_classification(
                    self, self.z_sum_left, self.z_sum_right,
                    &self.weighted_n_left, &self.weighted_n_right,
                    self.missing_go_to_left
                )
            else:
                for k in range(self.n_z_outputs):
                    memset(&self.z_sum_left[k, 0], 0, self.n_z_classes[k] * sizeof(float64_t))
                    memcpy(&self.z_sum_right[k, 0], &self.z_sum_total[k, 0],
                           self.n_z_classes[k] * sizeof(float64_t))
        
        return 0
    
    cdef int reverse_reset(self) except -1 nogil:
        """Reset to pos=end for both Y and Z."""
        cdef intp_t k
        
        ClassificationCriterion.reverse_reset(self)
        
        if self.n_z_outputs > 0:
            if self.n_missing != 0:
                _move_sums_classification(
                    self, self.z_sum_right, self.z_sum_left,
                    &self.weighted_n_right, &self.weighted_n_left,
                    not self.missing_go_to_left
                )
            else:
                for k in range(self.n_z_outputs):
                    memcpy(&self.z_sum_left[k, 0], &self.z_sum_total[k, 0],
                           self.n_z_classes[k] * sizeof(float64_t))
                    memset(&self.z_sum_right[k, 0], 0, self.n_z_classes[k] * sizeof(float64_t))
        
        return 0
    
    cdef int update(self, intp_t new_pos) except -1 nogil:
        """Update statistics when moving samples - OPTIMIZED like sklearn."""
        cdef intp_t old_pos, end_non_missing
        cdef const intp_t[:] sample_indices
        cdef const float64_t[:] sample_weight
        cdef intp_t i, p, k, c
        cdef float64_t w
        
        # Save old pos before parent updates it
        old_pos = self.pos
        end_non_missing = self.end - self.n_missing
        
        # First update Y statistics using parent's optimized method
        cdef int result = ClassificationCriterion.update(self, new_pos)
        if result != 0:
            return result
        
        # Update Z statistics - MATCH sklearn's optimization strategy
        if self.n_z_outputs > 0:
            sample_indices = self.sample_indices
            sample_weight = self.sample_weight
            w = 1.0
            
            # CRITICAL OPTIMIZATION: Choose shorter direction like sklearn does!
            # Update sum_left from the direction that requires fewer computations
            if (new_pos - old_pos) <= (end_non_missing - new_pos):
                # Forward direction: update from old_pos to new_pos (fewer samples)
                for p in range(old_pos, new_pos):
                    i = sample_indices[p]
                    
                    if sample_weight is not None:
                        w = sample_weight[i]
                    
                    for k in range(self.n_z_outputs):
                        self.z_sum_left[k, <intp_t> self.z[i, k]] += w
            else:
                # Backward direction: parent called reverse_reset, so sum_left = sum_total
                # Now subtract from new_pos to end to get correct sum_left
                for p in range(end_non_missing - 1, new_pos - 1, -1):
                    i = sample_indices[p]
                    
                    if sample_weight is not None:
                        w = sample_weight[i]
                    
                    for k in range(self.n_z_outputs):
                        self.z_sum_left[k, <intp_t> self.z[i, k]] -= w
            
            # CRITICAL OPTIMIZATION: Compute sum_right as difference instead of loop!
            # This matches sklearn line 540: sum_right = sum_total - sum_left
            for k in range(self.n_z_outputs):
                for c in range(self.n_z_classes[k]):
                    self.z_sum_right[k, c] = self.z_sum_total[k, c] - self.z_sum_left[k, c]
        
        return 0
    
    cdef float64_t z_node_impurity(self) noexcept nogil:
        """Calculate impurity for Z (implemented in subclasses)."""
        return 0.0
    
    cdef void z_children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil:
        """Calculate Z impurity for children (implemented in subclasses)."""
        impurity_left[0] = 0.0
        impurity_right[0] = 0.0
    
    cdef float64_t combined_node_impurity(self) noexcept nogil:
        """Compute combined impurity for current node: (1-theta)*Y + theta*Z.
        
        This represents the node's total impurity considering both objectives.
        
        OPTIMIZATIONS:
        - theta=0: Only compute Y (standard sklearn)
        - theta=1: Only compute Z (pure fairness)
        - 0<theta<1: Compute both and combine
        """
        cdef float64_t y_imp, z_imp
        
        # Fast path: theta=0, only Y matters
        if self.theta == 0.0 or self.n_z_outputs == 0:
            return self.node_impurity()
        
        # Fast path: theta=1, only Z matters
        if self.theta == 1.0:
            return self.z_node_impurity()
        
        # General case: combine both
        y_imp = self.node_impurity()
        z_imp = self.z_node_impurity()
        
        return (1.0 - self.theta) * y_imp + self.theta * z_imp
    
    cdef void combined_children_impurity(
        self,
        float64_t* combined_left,
        float64_t* combined_right
    ) noexcept nogil:
        """Compute combined impurity for children: (1-theta)*Y + theta*Z for each child.
        
        This represents each child's total impurity considering both objectives.
        
        OPTIMIZATIONS:
        - theta=0: Only compute Y (standard sklearn)
        - theta=1: Only compute Z (pure fairness)
        - 0<theta<1: Compute both and combine
        """
        cdef float64_t y_left, y_right, z_left, z_right
        
        # Fast path: theta=0, only Y matters
        if self.theta == 0.0 or self.n_z_outputs == 0:
            self.children_impurity(&y_left, &y_right)
            combined_left[0] = y_left
            combined_right[0] = y_right
            return
        
        # Fast path: theta=1, only Z matters
        if self.theta == 1.0:
            self.z_children_impurity(&z_left, &z_right)
            combined_left[0] = z_left
            combined_right[0] = z_right
            return
        
        # General case: combine both
        self.children_impurity(&y_left, &y_right)
        self.z_children_impurity(&z_left, &z_right)
        
        combined_left[0] = (1.0 - self.theta) * y_left + self.theta * z_left
        combined_right[0] = (1.0 - self.theta) * y_right + self.theta * z_right
    
    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        """Compute proxy impurity improvement with bias regularization - CORRECTED.
        
        The key insight: theta must be applied at the NODE LEVEL, not at the improvement level.
        
        Each node has a combined impurity: (1-theta)*Y_impurity + theta*Z_impurity
        
        The split improvement is: combined_parent - weighted_avg(combined_children)
        
        CRITICAL: Unlike standard sklearn, we MUST include the parent term because
        the combined_parent changes with theta. Different theta values mean different
        effective parent impurities, so we can't omit this "constant" term.
        
        If improvement is negative, the split makes things worse and shouldn't be taken.
        """
        cdef float64_t combined_parent, combined_left, combined_right
        cdef float64_t weighted_children
        
        # Get combined impurity of parent node
        combined_parent = self.combined_node_impurity()
        
        # Get combined impurities for children nodes
        self.combined_children_impurity(&combined_left, &combined_right)
        
        # Compute weighted children impurity
        weighted_children = (self.weighted_n_left * combined_left + 
                            self.weighted_n_right * combined_right)
        
        # Return improvement: parent_combined - weighted_children_combined
        # Positive = good split (reduction in combined impurity)
        # Negative = bad split (increase in combined impurity) - tree will reject it
        return self.weighted_n_node_samples * combined_parent - weighted_children
    
    cdef float64_t impurity_improvement(
        self,
        float64_t impurity_parent,
        float64_t impurity_left,
        float64_t impurity_right
    ) noexcept nogil:
        """Compute the actual improvement in combined impurity.
        
        NOTE: The input parameters impurity_parent, impurity_left, impurity_right
        are Y-only impurities passed by the tree builder. We ignore them and
        recompute using combined impurities.
        
        This method is called AFTER the best split is found to report the
        actual improvement value (used for pruning, feature importance, etc.).
        """
        cdef float64_t combined_parent, combined_left, combined_right
        
        # Get combined parent impurity
        combined_parent = self.combined_node_impurity()
        
        # Get combined children impurities
        self.combined_children_impurity(&combined_left, &combined_right)
        
        # Standard sklearn improvement formula using combined impurities
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (combined_parent - 
                 (self.weighted_n_right / self.weighted_n_node_samples * combined_right) -
                 (self.weighted_n_left / self.weighted_n_node_samples * combined_left)))



# =============================================================================
# AUC criterion (proportion-invariant, OvR for multi-class)
# =============================================================================
cdef inline float64_t _auc_from_split_counts(
    float64_t pos_left, float64_t pos_right,
    float64_t neg_left, float64_t neg_right,
    bint for_z=False
) noexcept nogil:
    cdef float64_t denom = (pos_left + pos_right) * (neg_left + neg_right)
    if denom <= 0.0:
        # For Y: return 0.5 (random baseline, no information)
        # For Z: return 1.0 (worst-case fairness violation)
        return 1.0 if for_z else 0.5
    return (pos_left * neg_right + 0.5 * (pos_left * neg_left + pos_right * neg_right)) / denom


cdef inline float64_t _aucz_transform(float64_t a) noexcept nogil:
    return fabs(a - 0.5) + 0.5


cdef class AUC(ClassificationCriterion):
    """
    Proportion-invariant split criterion based on AUC(y vs split_indicator).

    - For binary y: uses the single OvR class (class 1).
    - For multi-class y: computes OvR AUC per class (transformed to [0.5, 1.0])
      and averages across classes.
    - The splitter maximizes proxy_impurity_improvement(), which we define as:
        improvement = auc_split - 0.5
    """

    cdef inline bint _swap_children_by_ypos(self) noexcept nogil:
        """Return True if LEFT child should be treated as the positive-prediction
        side, based on the higher P(Y=1) among the two children.
        """
        cdef float64_t pos_left, pos_right, wL, wR
        if self.n_outputs <= 0:
            return False
        # Only defined for binary Y on output 0 (class index 1 == "positive")
        if self.n_classes[0] != 2:
            return False
        wL = self.weighted_n_left
        wR = self.weighted_n_right
        if wL <= 0.0 or wR <= 0.0:
            return False
        pos_left = self.sum_left[0, 1]
        pos_right = self.sum_right[0, 1]
        return (pos_left / wL) > (pos_right / wR)

    cdef inline float64_t _auc_y_split(self) noexcept nogil:
        cdef intp_t k, c
        cdef intp_t n_classes_k
        cdef float64_t auc_total = 0.0
        cdef float64_t auc_k, a
        cdef float64_t pos_left, pos_right, neg_left, neg_right

        for k in range(self.n_outputs):
            n_classes_k = self.n_classes[k]

            if n_classes_k <= 1:
                auc_total += 0.5
                continue

            if n_classes_k == 2:
                # Only need one OvR (class 1); class 0 is symmetric.
                pos_left = self.sum_left[k, 1]
                pos_right = self.sum_right[k, 1]
                neg_left = self.weighted_n_left - pos_left
                neg_right = self.weighted_n_right - pos_right
                if self._swap_children_by_ypos():
                    a = _auc_from_split_counts(pos_right, pos_left, neg_right, neg_left)
                else:
                    a = _auc_from_split_counts(pos_left, pos_right, neg_left, neg_right)
                auc_total += _aucz_transform(a)
            else:
                auc_k = 0.0
                for c in range(n_classes_k):
                    pos_left = self.sum_left[k, c]
                    pos_right = self.sum_right[k, c]
                    neg_left = self.weighted_n_left - pos_left
                    neg_right = self.weighted_n_right - pos_right
                    if self._swap_children_by_ypos():
                        a = _auc_from_split_counts(pos_right, pos_left, neg_right, neg_left)
                    else:
                        a = _auc_from_split_counts(pos_left, pos_right, neg_left, neg_right)
                    auc_k += _aucz_transform(a)
                auc_total += auc_k / n_classes_k

        return auc_total / self.n_outputs

    cdef float64_t node_impurity(self) noexcept nogil:
        # Not used for split selection (we override proxy_impurity_improvement).
        return 0.5

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        # Not used for split selection; keep consistent.
        impurity_left[0] = 0.5
        impurity_right[0] = 0.5

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        # Parent AUC is always 0.5 (constant for any node)
        # Children AUC is computed from the split
        # Gain = children - parent
        cdef float64_t parent_auc = 0.5
        cdef float64_t children_auc = self._auc_y_split()
        return children_auc - parent_auc

    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        # Keep consistent with proxy
        # Parent AUC is always 0.5, children AUC is from the split
        cdef float64_t parent_auc = 0.5
        cdef float64_t children_auc = self._auc_y_split()
        return children_auc - parent_auc


cdef class BiasedGini(BiasedClassificationCriterion):
    """Gini criterion with Z-based bias regularization.
    
    Combines Gini impurity for Y with Gini impurity for Z:
    - Minimizes Gini(Y) to find good splits for prediction
    - Maximizes Gini(Z) to maintain diversity in protected attributes
    """
    
    cdef float64_t node_impurity(self) noexcept nogil:
        """Gini impurity for Y only (used for reporting, not splitting)."""
        cdef float64_t gini = 0.0
        cdef float64_t sq_count
        cdef float64_t count_k
        cdef intp_t k, c
        
        for k in range(self.n_outputs):
            sq_count = 0.0
            for c in range(self.n_classes[k]):
                count_k = self.sum_total[k, c]
                sq_count += count_k * count_k
            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)
        
        return gini / self.n_outputs
    
    cdef void children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil:
        """Gini impurity for Y in left and right children."""
        cdef float64_t gini_left = 0.0
        cdef float64_t gini_right = 0.0
        cdef float64_t sq_count_left, sq_count_right
        cdef float64_t count_k
        cdef intp_t k, c
        
        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0
            
            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                sq_count_left += count_k * count_k
                
                count_k = self.sum_right[k, c]
                sq_count_right += count_k * count_k
            
            gini_left += 1.0 - sq_count_left / (self.weighted_n_left * self.weighted_n_left)
            gini_right += 1.0 - sq_count_right / (self.weighted_n_right * self.weighted_n_right)
        
        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs
    
    cdef float64_t z_node_impurity(self) noexcept nogil:
        """Gini impurity for Z (all columns aggregated) - OPTIMIZED."""
        # Declare all cdef variables at the top
        cdef float64_t p
        cdef float64_t gini_total
        cdef float64_t gini_k
        cdef float64_t sq_count
        cdef float64_t count_c
        cdef intp_t k, c
        cdef float64_t recip
        
        if self.n_z_outputs == 0:
            return 0.0
        
        # OPTIMIZATION: Fast path for single binary Z
        if self.n_z_outputs == 1 and self.n_z_classes[0] == 2:
            p = self.z_sum_total[0, 0] / self.weighted_n_node_samples
            return 2.0 * p * (1.0 - p)
        
        # General case
        gini_total = 0.0
        
        # OPTIMIZATION: Pre-compute reciprocal
        recip = 1.0 / (self.weighted_n_node_samples * self.weighted_n_node_samples)
        
        for k in range(self.n_z_outputs):
            sq_count = 0.0
            for c in range(self.n_z_classes[k]):
                count_c = self.z_sum_total[k, c]
                sq_count += count_c * count_c
            
            gini_k = 1.0 - sq_count * recip
            
            if self.Z_agg_mode == 0:  # mean
                gini_total += gini_k
            else:  # max (worst-case bias: take running MIN impurity)
                if k == 0 or gini_k < gini_total:
                    gini_total = gini_k
        
        if self.Z_agg_mode == 0:  # mean
            return gini_total / self.n_z_outputs
        else:  # max
            return gini_total
    
    cdef void z_children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil:
        """Gini impurity for Z in left and right children - OPTIMIZED."""
        # Declare all cdef variables at the top
        cdef float64_t p_left, p_right
        cdef float64_t gini_left_total, gini_right_total
        cdef float64_t gini_left_k, gini_right_k
        cdef float64_t sq_count_left, sq_count_right
        cdef float64_t count_c
        cdef intp_t k, c
        cdef float64_t recip_left, recip_right
        
        if self.n_z_outputs == 0:
            impurity_left[0] = 0.0
            impurity_right[0] = 0.0
            return
        
        # OPTIMIZATION 1: Fast path for single binary Z (most common case)
        # Binary Gini = 2 * p * (1-p) where p is proportion of class 0
        if self.n_z_outputs == 1 and self.n_z_classes[0] == 2:
            p_left = self.z_sum_left[0, 0] / self.weighted_n_left
            p_right = self.z_sum_right[0, 0] / self.weighted_n_right
            
            impurity_left[0] = 2.0 * p_left * (1.0 - p_left)
            impurity_right[0] = 2.0 * p_right * (1.0 - p_right)
            return
        
        # General case for multi-column or multi-class Z
        gini_left_total = 0.0
        gini_right_total = 0.0
        
        # OPTIMIZATION 2: Pre-compute reciprocals to avoid divisions in loop
        recip_left = 1.0 / (self.weighted_n_left * self.weighted_n_left)
        recip_right = 1.0 / (self.weighted_n_right * self.weighted_n_right)
        
        for k in range(self.n_z_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0
            
            for c in range(self.n_z_classes[k]):
                count_c = self.z_sum_left[k, c]
                sq_count_left += count_c * count_c
                
                count_c = self.z_sum_right[k, c]
                sq_count_right += count_c * count_c
            
            # Use multiplication instead of division
            gini_left_k = 1.0 - sq_count_left * recip_left
            gini_right_k = 1.0 - sq_count_right * recip_right
            
            if self.Z_agg_mode == 0:  # mean
                gini_left_total += gini_left_k
                gini_right_total += gini_right_k
            else:  # max (worst-case bias: take running MIN impurity)
                if k == 0 or gini_left_k < gini_left_total:
                    gini_left_total = gini_left_k
                if k == 0 or gini_right_k < gini_right_total:
                    gini_right_total = gini_right_k
        
        if self.Z_agg_mode == 0:  # mean
            impurity_left[0] = gini_left_total / self.n_z_outputs
            impurity_right[0] = gini_right_total / self.n_z_outputs
        else:  # max
            impurity_left[0] = gini_left_total
            impurity_right[0] = gini_right_total


cdef class BiasedEntropy(BiasedClassificationCriterion):
    """Entropy criterion with Z-based bias regularization.
    
    Combines Entropy for Y with Entropy for Z:
    - Minimizes Entropy(Y) to find good splits for prediction
    - Maximizes Entropy(Z) to maintain diversity in protected attributes
    """
    
    cdef float64_t node_impurity(self) noexcept nogil:
        """Entropy for Y only."""
        cdef float64_t entropy = 0.0
        cdef float64_t count_k
        cdef intp_t k, c
        
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_total[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)
        
        return entropy / self.n_outputs
    
    cdef void children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil:
        """Entropy for Y in left and right children."""
        cdef float64_t entropy_left = 0.0
        cdef float64_t entropy_right = 0.0
        cdef float64_t count_k
        cdef intp_t k, c
        
        for k in range(self.n_outputs):
            for c in range(self.n_classes[k]):
                count_k = self.sum_left[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)
                
                count_k = self.sum_right[k, c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)
        
        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs
    
    cdef float64_t z_node_impurity(self) noexcept nogil:
        """Entropy for Z (all columns aggregated)."""
        if self.n_z_outputs == 0:
            return 0.0
        
        cdef float64_t entropy_total = 0.0
        cdef float64_t entropy_k
        cdef float64_t count_c
        cdef intp_t k, c
        
        for k in range(self.n_z_outputs):
            entropy_k = 0.0
            for c in range(self.n_z_classes[k]):
                count_c = self.z_sum_total[k, c]
                if count_c > 0.0:
                    count_c /= self.weighted_n_node_samples
                    entropy_k -= count_c * log(count_c)
            
            if self.Z_agg_mode == 0:  # mean
                entropy_total += entropy_k
            else:  # max (worst-case bias: take running MIN impurity)
                if k == 0 or entropy_k < entropy_total:
                    entropy_total = entropy_k
        
        if self.Z_agg_mode == 0:  # mean
            return entropy_total / self.n_z_outputs
        else:  # max
            return entropy_total
    
    cdef void z_children_impurity(
        self,
        float64_t* impurity_left,
        float64_t* impurity_right
    ) noexcept nogil:
        """Entropy for Z in left and right children - OPTIMIZED."""
        # Declare all cdef variables at the top
        cdef float64_t p_left, p_right
        cdef float64_t entropy_left_total, entropy_right_total
        cdef float64_t entropy_left_k, entropy_right_k
        cdef float64_t count_c
        cdef intp_t k, c
        cdef float64_t recip_left, recip_right
        
        if self.n_z_outputs == 0:
            impurity_left[0] = 0.0
            impurity_right[0] = 0.0
            return
        
        # OPTIMIZATION 1: Fast path for single binary Z
        if self.n_z_outputs == 1 and self.n_z_classes[0] == 2:
            p_left = self.z_sum_left[0, 0] / self.weighted_n_left
            p_right = self.z_sum_right[0, 0] / self.weighted_n_right
            
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            # Using xlogy for numerical stability (handles p=0 case)
            impurity_left[0] = -(xlogy(p_left, p_left) + xlogy(1.0 - p_left, 1.0 - p_left))
            impurity_right[0] = -(xlogy(p_right, p_right) + xlogy(1.0 - p_right, 1.0 - p_right))
            return
        
        # General case
        entropy_left_total = 0.0
        entropy_right_total = 0.0
        
        # OPTIMIZATION 2: Pre-compute reciprocals
        recip_left = 1.0 / self.weighted_n_left
        recip_right = 1.0 / self.weighted_n_right
        
        for k in range(self.n_z_outputs):
            entropy_left_k = 0.0
            entropy_right_k = 0.0
            
            for c in range(self.n_z_classes[k]):
                count_c = self.z_sum_left[k, c]
                if count_c > 0.0:
                    count_c *= recip_left  # Use multiplication instead of division
                    entropy_left_k -= count_c * log(count_c)
                
                count_c = self.z_sum_right[k, c]
                if count_c > 0.0:
                    count_c *= recip_right  # Use multiplication instead of division
                    entropy_right_k -= count_c * log(count_c)
            
            if self.Z_agg_mode == 0:  # mean
                entropy_left_total += entropy_left_k
                entropy_right_total += entropy_right_k
            else:  # max (worst-case bias: take running MIN impurity)
                if k == 0 or entropy_left_k < entropy_left_total:
                    entropy_left_total = entropy_left_k
                if k == 0 or entropy_right_k < entropy_right_total:
                    entropy_right_total = entropy_right_k
        
        if self.Z_agg_mode == 0:  # mean
            impurity_left[0] = entropy_left_total / self.n_z_outputs
            impurity_right[0] = entropy_right_total / self.n_z_outputs
        else:  # max
            impurity_left[0] = entropy_left_total
            impurity_right[0] = entropy_right_total


# =============================================================================
# Factory functions to create biased criteria (workaround for __cinit__ issues)
# =============================================================================


# =============================================================================
# Biased AUC criterion: score = (1-theta)*AUC_y - theta*AUC_z
# where:
#   AUC_y = AUC(y vs split_indicator) transformed to [0.5, 1.0]
#   AUC_z = worst-case (or mean) AUC(z_g=value vs split_indicator) transformed to [0.5, 1.0]
#
# We define the improvement used by the splitter as:
#   parent_score = (1-theta)*0.5 - theta*0.5
#   score        = (1-theta)*AUC_y - theta*AUC_z
#   improvement  = score - parent_score
#
# This makes theta=1 focus purely on minimizing AUC_z (fairness), since it maximizes (-AUC_z).
# =============================================================================
cdef class BiasedAUC(BiasedClassificationCriterion):

    cdef inline bint _swap_children_by_ypos(self) noexcept nogil:
        """Return True if LEFT child should be treated as the positive-prediction
        side, based on the higher P(Y=1) among the two children.

        This swap decision is then applied consistently when computing both AUC_y
        and AUC_z, so that AUC_z measures leakage under the *same* predictions
        induced by the split (predictions are defined by Y, not by Z).
        """
        cdef float64_t pos_left, pos_right, wL, wR
        if self.n_outputs <= 0:
            return False
        # Only defined for binary Y on output 0 (class index 1 == "positive")
        if self.n_classes[0] != 2:
            return False
        wL = self.weighted_n_left
        wR = self.weighted_n_right
        if wL <= 0.0 or wR <= 0.0:
            return False
        pos_left = self.sum_left[0, 1]
        pos_right = self.sum_right[0, 1]
        return (pos_left / wL) > (pos_right / wR)


    cdef inline float64_t _auc_y_split(self) noexcept nogil:
        cdef intp_t k, c
        cdef intp_t n_classes_k
        cdef float64_t auc_total = 0.0
        cdef float64_t auc_k, a
        cdef float64_t pos_left, pos_right, neg_left, neg_right

        for k in range(self.n_outputs):
            n_classes_k = self.n_classes[k]

            if n_classes_k <= 1:
                auc_total += 0.5
                continue

            if n_classes_k == 2:
                pos_left = self.sum_left[k, 1]
                pos_right = self.sum_right[k, 1]
                neg_left = self.weighted_n_left - pos_left
                neg_right = self.weighted_n_right - pos_right

                # Predictions are defined by which child has higher P(Y=1).
                # _auc_from_split_counts assumes LEFT has score 0 and RIGHT has score 1.
                # If LEFT is the positive-prediction side, swap L/R so that the
                # positive-prediction side maps to RIGHT (score 1) consistently.
                if self._swap_children_by_ypos():
                    a = _auc_from_split_counts(pos_right, pos_left, neg_right, neg_left)
                else:
                    a = _auc_from_split_counts(pos_left, pos_right, neg_left, neg_right)

                auc_total += _aucz_transform(a)
            else:
                auc_k = 0.0
                for c in range(n_classes_k):
                    pos_left = self.sum_left[k, c]
                    pos_right = self.sum_right[k, c]
                    neg_left = self.weighted_n_left - pos_left
                    neg_right = self.weighted_n_right - pos_right
                    if self._swap_children_by_ypos():
                        a = _auc_from_split_counts(pos_right, pos_left, neg_right, neg_left)
                    else:
                        a = _auc_from_split_counts(pos_left, pos_right, neg_left, neg_right)
                    auc_k += _aucz_transform(a)
                auc_total += auc_k / n_classes_k

        return auc_total / self.n_outputs

    cdef inline float64_t _auc_z_split(self) noexcept nogil:
        cdef intp_t g, c
        cdef intp_t n_classes_g
        cdef float64_t a, auc_g, auc_total
        cdef float64_t pos_left, pos_right, neg_left, neg_right

        if self.n_z_outputs <= 0:
            return 0.5

        if self.Z_agg_mode == 0:
            auc_total = 0.0
        else:
            auc_total = 0.5  # for max

        for g in range(self.n_z_outputs):
            n_classes_g = self.n_z_classes[g]
           
            if n_classes_g <= 1:
                # Only one group → can't measure disparity → worst-case
                auc_g = 1.0
            elif n_classes_g == 2:
                pos_left = self.z_sum_left[g, 1]
                pos_right = self.z_sum_right[g, 1]
                neg_left = self.weighted_n_left - pos_left
                neg_right = self.weighted_n_right - pos_right
                if self._swap_children_by_ypos():
                    a = _auc_from_split_counts(pos_right, pos_left, neg_right, neg_left, for_z=True)
                else:
                    a = _auc_from_split_counts(pos_left, pos_right, neg_left, neg_right, for_z=True)
                auc_g = _aucz_transform(a)
            else:
                # Multi-class: aggregate across classes within this attribute
                if self.Z_agg_mode == 0:  # mean
                    auc_g = 0.0
                else:  # max
                    auc_g = 0.5
                    
                for c in range(n_classes_g):
                    pos_left = self.z_sum_left[g, c]
                    pos_right = self.z_sum_right[g, c]
                    neg_left = self.weighted_n_left - pos_left
                    neg_right = self.weighted_n_right - pos_right
                    if self._swap_children_by_ypos():
                        a = _auc_from_split_counts(pos_right, pos_left, neg_right, neg_left, for_z=True)
                    else:
                        a = _auc_from_split_counts(pos_left, pos_right, neg_left, neg_right, for_z=True)
                    a = _aucz_transform(a)
                    
                    if self.Z_agg_mode == 0:  # mean across classes
                        auc_g += a
                    else:  # max across classes
                        if a > auc_g:
                            auc_g = a
                            
                if self.Z_agg_mode == 0:  # finalize mean across classes
                    auc_g /= n_classes_g

            # Aggregate across attributes
            if self.Z_agg_mode == 0:  # mean across attributes
                auc_total += auc_g
            else:  # max across attributes (worst-case)
                if g == 0 or auc_g > auc_total:
                    auc_total = auc_g
                    
        if self.Z_agg_mode == 0:
            auc_total /= self.n_z_outputs

        return auc_total

    cdef float64_t node_impurity(self) noexcept nogil:
        # Return non-zero value to allow split evaluation
        # (tree builder checks if impurity==0 and skips splitting)
        return 0.5

    cdef void children_impurity(self, float64_t* impurity_left,
                                float64_t* impurity_right) noexcept nogil:
        # Return non-zero values (matching standard AUC)
        impurity_left[0] = 0.5
        impurity_right[0] = 0.5

    cdef float64_t proxy_impurity_improvement(self) noexcept nogil:
        cdef float64_t auc_y = self._auc_y_split()
        cdef float64_t auc_z = self._auc_z_split()
        cdef float64_t parent = ((1.0 - self.theta) * 0.5) - (self.theta * 0.5)
        cdef float64_t score = ((1.0 - self.theta) * auc_y) - (self.theta * auc_z)
        cdef float64_t improvement = score - parent
        
        return improvement
    
	
    cdef float64_t impurity_improvement(self, float64_t impurity_parent,
                                        float64_t impurity_left,
                                        float64_t impurity_right) noexcept nogil:
        return self.proxy_impurity_improvement()


def create_biased_auc(
    intp_t n_outputs,
    cnp.ndarray[intp_t, ndim=1] n_classes,
    const float64_t[:, ::1] z,
    cnp.ndarray[intp_t, ndim=1] n_z_classes,
    float64_t theta,
    int Z_agg_mode,
):
    """Factory function to create BiasedAUC criterion.

    Mirrors create_biased_gini/create_biased_entropy to ensure that theta and
    Z-related statistics buffers are correctly initialised (avoids __cinit__
    inheritance issues).
    """
    cdef BiasedAUC criterion = BiasedAUC(n_outputs, n_classes)

    cdef intp_t k
    cdef intp_t max_n_z_classes

    criterion.z = z
    criterion.n_z_outputs = z.shape[1] if z.shape[1] > 0 else 0
    criterion.theta = theta
    criterion.Z_agg_mode = Z_agg_mode

    if criterion.n_z_outputs > 0:
        criterion.n_z_classes = np.empty(criterion.n_z_outputs, dtype=np.intp)
        max_n_z_classes = 0

        for k in range(criterion.n_z_outputs):
            criterion.n_z_classes[k] = n_z_classes[k]
            if n_z_classes[k] > max_n_z_classes:
                max_n_z_classes = n_z_classes[k]

        criterion.max_n_z_classes = max_n_z_classes

        criterion.z_sum_total = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
        criterion.z_sum_left  = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
        criterion.z_sum_right = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
    else:
        criterion.max_n_z_classes = 0

    return criterion

def create_biased_gini(intp_t n_outputs,
                       cnp.ndarray[intp_t, ndim=1] n_classes,
                       const float64_t[:, ::1] z,
                       cnp.ndarray[intp_t, ndim=1] n_z_classes,
                       float64_t theta,
                       int Z_agg_mode):
    """Factory function to create BiasedGini criterion.
    
    This avoids Cython's __cinit__ inheritance issues.
    """
    # Create using parent constructor (which will call parent __cinit__)
    cdef BiasedGini criterion = BiasedGini(n_outputs, n_classes)
    
    # Now initialize Z-related fields
    cdef intp_t k
    cdef intp_t max_n_z_classes
    
    criterion.z = z
    criterion.n_z_outputs = z.shape[1] if z.shape[1] > 0 else 0
    criterion.theta = theta
    criterion.Z_agg_mode = Z_agg_mode
    
    if criterion.n_z_outputs > 0:
        criterion.n_z_classes = np.empty(criterion.n_z_outputs, dtype=np.intp)
        max_n_z_classes = 0
        
        for k in range(criterion.n_z_outputs):
            criterion.n_z_classes[k] = n_z_classes[k]
            if n_z_classes[k] > max_n_z_classes:
                max_n_z_classes = n_z_classes[k]
        
        criterion.max_n_z_classes = max_n_z_classes
        
        # Allocate Z statistics arrays
        criterion.z_sum_total = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
        criterion.z_sum_left = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
        criterion.z_sum_right = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
    else:
        criterion.max_n_z_classes = 0
    
    return criterion


def create_biased_entropy(intp_t n_outputs,
                          cnp.ndarray[intp_t, ndim=1] n_classes,
                          const float64_t[:, ::1] z,
                          cnp.ndarray[intp_t, ndim=1] n_z_classes,
                          float64_t theta,
                          int Z_agg_mode):
    """Factory function to create BiasedEntropy criterion.
    
    This avoids Cython's __cinit__ inheritance issues.
    """
    # Create using parent constructor (which will call parent __cinit__)
    cdef BiasedEntropy criterion = BiasedEntropy(n_outputs, n_classes)
    
    # Now initialize Z-related fields
    cdef intp_t k
    cdef intp_t max_n_z_classes
    
    criterion.z = z
    criterion.n_z_outputs = z.shape[1] if z.shape[1] > 0 else 0
    criterion.theta = theta
    criterion.Z_agg_mode = Z_agg_mode
    
    if criterion.n_z_outputs > 0:
        criterion.n_z_classes = np.empty(criterion.n_z_outputs, dtype=np.intp)
        max_n_z_classes = 0
        
        for k in range(criterion.n_z_outputs):
            criterion.n_z_classes[k] = n_z_classes[k]
            if n_z_classes[k] > max_n_z_classes:
                max_n_z_classes = n_z_classes[k]
        
        criterion.max_n_z_classes = max_n_z_classes
        
        # Allocate Z statistics arrays
        criterion.z_sum_total = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
        criterion.z_sum_left = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
        criterion.z_sum_right = np.zeros((criterion.n_z_outputs, max_n_z_classes), dtype=np.float64)
    else:
        criterion.max_n_z_classes = 0
    
    return criterion

