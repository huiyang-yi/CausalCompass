"""Regression checks for the 0.2.0 public API fixes (no model training)."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score

from causalcompass.algorithms import VARLiNGAM, CUTSPlus
from causalcompass.algorithms.cuts.utils import data_interpolate
from causalcompass.datasets import missing


class ReleaseRegressionTests(unittest.TestCase):
    def test_varlingam_edge_direction_and_single_fit_sweep(self):
        model = VARLiNGAM(varlingamalpha=0.5)
        scores = np.zeros((3, 3))
        scores[1, 0] = 0.9  # X_0 causes X_1
        expected = np.zeros((3, 3), dtype=int)
        expected[1, 0] = 1
        with patch.object(model, 'run_raw', return_value={'score_matrix': scores}) as raw:
            np.testing.assert_array_equal(model.run(np.zeros((20, 3))), expected)
            raw.reset_mock()
            results = model.run_threshold_sweep(np.zeros((20, 3)), [0.5, 0.95])
            raw.assert_called_once()
            np.testing.assert_array_equal(results[0.5], expected)
            np.testing.assert_array_equal(results[0.95], np.zeros((3, 3)))

    def test_cutsplus_nshd_and_unchanged_ranking_metrics(self):
        truth = np.zeros((3, 3), dtype=int)
        truth[1, 0] = 1
        logits = np.array([[-6., -4., -3.], [2., -5., -2.], [-1., -.5, -4.]])
        model = CUTSPlus(device='cpu')
        for thresholds in (None, [0.5]):
            results = model.eval(truth, logits, shd_thresholds=thresholds)
            for metrics, mask in zip(results, (np.ones((3, 3), bool), ~np.eye(3, dtype=bool))):
                self.assertEqual(metrics['shd'], 0.)
                self.assertEqual(metrics['auroc'], roc_auc_score(truth[mask], logits[mask]))
                self.assertEqual(metrics['auprc'], average_precision_score(truth[mask], logits[mask]))

    def test_cutsplus_sigmoid_applied_once_and_diagonal_exclusion(self):
        truth = np.zeros((3, 3), dtype=int)
        truth[1, 0] = 1
        logits = np.full((3, 3), -4.)
        logits[1, 0] = 2.
        # expit(2) > .8, but expit(expit(2)) < .8.
        model = CUTSPlus(device='cpu')
        a, b = model.eval(truth, logits, shd_thresholds=[0.8])
        self.assertEqual(a['shd'], 0.)
        self.assertEqual(b['shd'], 0.)
        logits[0, 0] = 1000.
        logits[2, 1] = -1000.
        a, b = model.eval(truth, logits, shd_thresholds=[0.8])
        self.assertEqual(a['shd'], 1.)
        self.assertEqual(b['shd'], 0.)
        self.assertTrue(np.isfinite(expit(logits)).all())

    def test_gp_rejected_by_public_and_bundled_helpers(self):
        data = np.arange(12.).reshape(6, 2)
        mask = np.ones_like(data)
        for method in ('GP', 'gp', 'Gp'):
            with self.assertRaises(ValueError):
                missing.normalize_missing_imputation_method(method)
            with self.assertRaises(ValueError):
                missing.interp_multivar_data(data, mask, interp=method)
            with self.assertRaises(ValueError):
                data_interpolate.interp_multivar_data(data[..., None], mask[..., None], interp=method)
            with self.assertRaises(ValueError):
                data_interpolate.interp_masked_data(data, mask[:, :1], interp=method)

    def test_supported_imputations_keep_original_data_and_mask(self):
        kwargs = dict(p=10, T=80, seed=4, burn_in=30,
                      missing_config={'random_missing': {'missing_prob': .3, 'missing_var': 'all'}})
        outputs = [missing.generate_missing_var(interp=m, **kwargs) for m in ('zoh', 'linear')]
        for out in outputs:
            self.assertEqual(out[0].shape, (80, 10, 1))
            self.assertTrue(np.isfinite(out[0]).all())
        np.testing.assert_array_equal(outputs[0][2], outputs[1][2])
        np.testing.assert_array_equal(outputs[0][4], outputs[1][4])

    def test_batch_missing_probability_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(dataset_type='VAR', p=10, T=80, seed=4,
                          missing_prob=.3, output_dir=directory, burn_in=30)
            paths = missing.generate_and_save_missing_dataset_variants(**kwargs)
            self.assertEqual(set(paths), {'zoh', 'linear'})
            with np.load(paths['zoh']) as zoh, np.load(paths['linear']) as linear:
                self.assertTrue(np.any(zoh['mask'] == 0))
                self.assertTrue(np.any(zoh['mask'] == 1))
                self.assertLess(abs(np.mean(zoh['mask'] == 0) - .3), .1)
                np.testing.assert_array_equal(zoh['mask'], linear['mask'])
                np.testing.assert_array_equal(zoh['original_data'], linear['original_data'])
            paths = missing.generate_and_save_missing_dataset_variants(
                **kwargs, missing_config={'random_missing': {'missing_prob': 0., 'missing_var': 'all'}})
            with np.load(paths['zoh']) as data:
                self.assertTrue(np.all(data['mask'] == 1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
