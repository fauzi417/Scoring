import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection._base import SelectorMixin
from sklearn.utils.sparsefuncs import mean_variance_axis, min_max_axis
from sklearn.utils.validation import check_is_fitted
from sklearn.utils._tags import _safe_tags
from sklearn.utils import (
    check_array,
    safe_mask,
    safe_sqr,
)


class ReverseVarianceThreshold(SelectorMixin, BaseEstimator):

    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y=None):

        X = self._validate_data(
            X,
            accept_sparse=("csr", "csc"),
            dtype=np.float64,
            force_all_finite="allow-nan",
        )

        if hasattr(X, "toarray"):  # sparse matrix
            _, self.variances_ = mean_variance_axis(X, axis=0)
            if self.threshold == 0:
                mins, maxes = min_max_axis(X, axis=0)
                peak_to_peaks = maxes - mins
        else:
            self.variances_ = np.nanvar(X, axis=0)
            if self.threshold == 0:
                peak_to_peaks = np.ptp(X, axis=0)

        if self.threshold == 0:
            compare_arr = np.array([self.variances_, peak_to_peaks])
            self.variances_ = np.nanmin(compare_arr, axis=0)
        elif self.threshold < 0.0:
            raise ValueError(f"Threshold must be non-negative. Got: {self.threshold}")

        if np.all(~np.isfinite(self.variances_) | (self.variances_ <= self.threshold)):
            msg = "No feature in X meets the variance threshold {0:.5f}"
            if X.shape[0] == 1:
                msg += " (X contains only one sample)"
            raise ValueError(msg.format(self.threshold))

        return self

    def _get_support_mask(self):
        check_is_fitted(self)

        return self.variances_ < self.threshold


    def transform(self, X):
        X = self._validate_data(
            X,
            dtype=None,
            accept_sparse="csr",
            force_all_finite=not _safe_tags(self, key="allow_nan"),
            reset=False,
        )
        return self._transform(X)

    def _transform(self, X):
        mask = self.get_support()
        if not mask.any():
            warn(
                "No features were selected: either the data is"
                " too noisy or the selection test too strict.",
                UserWarning,
            )
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return X[:, safe_mask(X, mask)]
