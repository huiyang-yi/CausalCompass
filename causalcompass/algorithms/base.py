import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

class BaseCausalAlgorithm:
    """
    Base class for all causal discovery algorithms.

    Methods
    -------
    __init__(seed=None)
        Initialize with a random seed for reproducibility.
    run(X)
        Run the algorithm on input data X of shape (T, p). Returns a predicted adjacency matrix of shape (p, p).
    eval(true_adj, predicted_adj, shd_thresholds=None)
        Evaluate the predicted adjacency matrix against the ground truth. Returns
        (all_metrics, no_diag_metrics), where each metric dict contains 'auroc',
        'auprc', and 'shd'. For continuous-output algorithms, SHD uses
        abs(predicted_adj) and either a user-provided threshold list or a
        TCD-Arena-style automatic threshold search.
    """
    def __init__(self, seed=None, **kwargs):
        self.seed = seed
        self.params = kwargs

    def run(self, X, **kwargs):
        raise NotImplementedError("Subclasses must implement run()")

    def run_raw(self, X, **kwargs):
        """
        Run the algorithm and return an unthresholded intermediate result that can
        be reused across multiple threshold values.
        """
        raise NotImplementedError("Subclasses must implement run_raw()")

    def _threshold_from_raw(self, raw_result, threshold):
        """
        Convert the output of ``run_raw`` into a thresholded adjacency matrix.
        """
        raise NotImplementedError("Subclasses must implement _threshold_from_raw()")

    @staticmethod
    def _coerce_thresholds(thresholds):
        if np.isscalar(thresholds):
            return [float(thresholds)]

        return [float(threshold) for threshold in thresholds]

    def run_threshold_sweep(self, X, thresholds):
        """
        Run the algorithm once and post-process the raw result for each threshold.
        """
        raw_result = self.run_raw(X)
        return {
            threshold: self._threshold_from_raw(raw_result, threshold)
            for threshold in self._coerce_thresholds(thresholds)
        }

    def _is_continuous_output(self):
        return getattr(self, "_eval_output_type", "binary") == "continuous"

    @staticmethod
    def _compute_binary_shd(y_true, y_pred_binary):
        """
        Compute normalized SHD for binary predictions.

        When there are no positive edges in y_true, return 0.0 only if the
        prediction is also all-zero; otherwise return the raw mismatch count to
        avoid divide-by-zero while still penalizing false positives.
        """
        y_true = np.asarray(y_true).astype(int).reshape(-1)
        y_pred_binary = np.asarray(y_pred_binary).astype(int).reshape(-1)

        mismatches = float(np.sum(y_pred_binary != y_true))
        positive_edges = int(np.sum(y_true))
        if positive_edges == 0:
            return 0.0 if mismatches == 0 else mismatches
        return mismatches / positive_edges

    @staticmethod
    def _default_shd_thresholds(abs_scores):
        abs_scores = np.asarray(abs_scores, dtype=float).reshape(-1)
        if abs_scores.size == 0:
            return np.array([0.0], dtype=float)

        score_max = float(abs_scores.max())
        if float(abs_scores.min()) == score_max:
            return np.array([0.0, score_max + 1e-6], dtype=float)

        thresholds = np.linspace(float(abs_scores.min()), score_max, num=100)
        return np.concatenate((
            np.array([0.0], dtype=float),
            thresholds,
            np.array([score_max + 1e-6], dtype=float),
        ))

    def _compute_continuous_shd(self, y_true, y_scores, shd_thresholds=None):
        y_true = np.asarray(y_true).astype(int).reshape(-1)
        abs_scores = np.abs(np.asarray(y_scores, dtype=float).reshape(-1))

        if shd_thresholds is None:
            thresholds = self._default_shd_thresholds(abs_scores)
        else:
            thresholds = np.asarray(list(shd_thresholds), dtype=float)
            if thresholds.ndim != 1 or thresholds.size == 0:
                raise ValueError("shd_thresholds must be a non-empty 1D iterable of thresholds.")

        shd_values = [
            self._compute_binary_shd(y_true, abs_scores > threshold)
            for threshold in thresholds
        ]
        return float(np.min(shd_values))

    def eval(self, true_adj, predicted_adj, shd_thresholds=None):
        """
        Evaluate the predicted adjacency matrix against the ground truth.

        Parameters
        ----------
        true_adj : np.ndarray
            Ground-truth adjacency matrix.
        predicted_adj : np.ndarray
            Predicted adjacency matrix or score matrix.
        shd_thresholds : iterable of float, optional
            Candidate thresholds used only for continuous-output algorithms when
            computing SHD. If None, a TCD-Arena-style automatic threshold search
            is used on abs(predicted_adj).

        Returns
        -------
        tuple[dict, dict]
            (all_metrics, no_diag_metrics), where each dict contains 'auroc',
            'auprc', and 'shd'.
        """
        true_adj = np.asarray(true_adj)
        predicted_adj = np.asarray(predicted_adj)
        if true_adj.shape != predicted_adj.shape:
            raise ValueError("true_adj and predicted_adj must have the same shape.")

        # Original metrics (including self-loops)
        y_true_all = true_adj.flatten()
        y_scores_all = predicted_adj.flatten()

        auroc_all = roc_auc_score(y_true_all, y_scores_all)
        auprc_all = average_precision_score(y_true_all, y_scores_all)

        # Metrics without diagonal elements (excluding self-loops)
        p = true_adj.shape[0]
        mask = ~np.eye(p, dtype=bool)  # Create mask for non-diagonal elements

        y_true_no_diag = true_adj[mask]
        y_scores_no_diag = predicted_adj[mask]

        auroc_no_diag = roc_auc_score(y_true_no_diag, y_scores_no_diag)
        auprc_no_diag = average_precision_score(y_true_no_diag, y_scores_no_diag)

        if self._is_continuous_output():
            shd_all = self._compute_continuous_shd(y_true_all, y_scores_all, shd_thresholds=shd_thresholds)
            shd_no_diag = self._compute_continuous_shd(y_true_no_diag, y_scores_no_diag, shd_thresholds=shd_thresholds)
        else:
            shd_all = self._compute_binary_shd(y_true_all, y_scores_all)
            shd_no_diag = self._compute_binary_shd(y_true_no_diag, y_scores_no_diag)

        all_metrics = {
            'auroc': auroc_all,
            'auprc': auprc_all,
            'shd': shd_all,
        }

        no_diag_metrics = {
            'auroc': auroc_no_diag,
            'auprc': auprc_no_diag,
            'shd': shd_no_diag,
        }

        return all_metrics, no_diag_metrics
