from __future__ import annotations

import math
import os
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import torch

from manifold_repsim.metrics import AlignmentMetrics
from manifold_repsim.resi import adapter
from manifold_repsim.resi.measures import (
    RESI_MEASURE_SPECS,
    get_measure_spec,
    manifold_measure_categories,
)
from manifold_repsim.resi.campaign.schema import LOCAL_MANIFOLD_MEASURES
from manifold_repsim.resi.analysis.config import measure_category


MEASURE_CLASSES = [
    getattr(adapter, name) for name in adapter.MANIFOLD_RESI_MEASURE_CLASSES
]


class TestReSiManifoldAdapter(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.R = rng.randn(12, 5).astype(np.float32)
        self.Rp = (self.R + 0.05 * rng.randn(12, 5)).astype(np.float32)

    def test_all_registered_classes_return_finite_scores(self):
        with mock.patch.dict(
            os.environ, {"MANIFOLD_RESI_AUC_SWEEP_LEN": "4"}, clear=False
        ):
            for measure_class in MEASURE_CLASSES:
                with self.subTest(measure=measure_class.__name__):
                    value = measure_class()(self.R, self.Rp, "nd")
                    self.assertIsInstance(value, float)
                    self.assertTrue(math.isfinite(value))

    def test_fixed_knn_clamps_topk_for_small_inputs(self):
        for measure_class in (
            adapter.MutualKNNTop10,
            adapter.CKNNATop10,
        ):
            self.assertTrue(
                math.isfinite(measure_class()(self.R[:5], self.Rp[:5], "nd"))
            )

    def test_metrics_allow_matching_samples_with_different_feature_widths(self):
        rng = np.random.RandomState(1)
        wider = rng.randn(len(self.R), 7).astype(np.float32)
        with mock.patch.dict(
            os.environ,
            {"MANIFOLD_RESI_AUC_SWEEP_LEN": "4", "MANIFOLD_RESI_MAX_POINTS": ""},
            clear=False,
        ):
            for measure_class in MEASURE_CLASSES:
                with self.subTest(measure=measure_class.__name__):
                    self.assertTrue(math.isfinite(measure_class()(self.R, wider, "nd")))

    def test_device_is_unset_by_default_and_leaves_representations_in_place(self):
        env = {"MANIFOLD_RESI_MAX_POINTS": "", "MANIFOLD_RESI_DEVICE": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertIsNone(adapter._resolve_device())
            source = torch.as_tensor(self.R)
            moved, _, info = adapter._move_pair(source, source, None)
            self.assertIs(moved, source)
            self.assertIsNone(info["device"])

    def test_cpu_device_round_trips_and_is_recorded_in_the_signal(self):
        """A configured device must reach the metric and be visible in provenance."""
        env = {
            "MANIFOLD_RESI_MAX_POINTS": "",
            "MANIFOLD_RESI_DEVICE": "cpu",
            "MANIFOLD_RESI_AUC_SWEEP_LEN": "4",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(adapter._resolve_device(), torch.device("cpu"))
            with mock.patch.object(adapter, "prepare_metric_pair") as prepare:
                with mock.patch.object(
                    adapter, "score_prepared_metric", return_value=0.5
                ):
                    adapter.CKArbfSigma05()(self.R, self.Rp, "nd")
            self.assertEqual(prepare.call_args.args[1].device, torch.device("cpu"))

            auc = adapter.CKArbfAUC()
            self.assertTrue(math.isfinite(auc(self.R, self.Rp, "nd")))
            self.assertEqual(auc.last_similarity_signal["device"], "cpu")

    def test_cuda_request_without_cuda_fails_loudly(self):
        """Silently falling back to CPU would waste exactly what the flag avoids."""
        env = {"MANIFOLD_RESI_DEVICE": "cuda"}
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "no CUDA device"):
                    adapter._resolve_device()

    def test_metrics_reject_mismatched_sample_counts(self):
        with self.assertRaisesRegex(ValueError, "mismatched sample counts"):
            adapter.MutualKNNTop10()(self.R, self.Rp[:-1], "nd")

    def test_general_point_limit_applies_to_fixed_and_auc_metrics(self):
        rng = np.random.RandomState(2)
        wider = rng.randn(len(self.R), 7).astype(np.float32)
        env = {
            "MANIFOLD_RESI_MAX_POINTS": "7",
            "MANIFOLD_RESI_AUC_MAX_POINTS": "",
            "MANIFOLD_RESI_SUBSAMPLE_SEED": "3",
            "MANIFOLD_RESI_AUC_SWEEP_LEN": "4",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch.object(adapter, "prepare_metric_pair") as prepare:
                with mock.patch.object(
                    adapter, "score_prepared_metric", return_value=0.5
                ):
                    self.assertEqual(adapter.CKArbfSigma05()(self.R, wider, "nd"), 0.5)
            self.assertEqual(prepare.call_args.args[1].shape[0], 7)
            self.assertEqual(prepare.call_args.args[2].shape[0], 7)

            auc = adapter.MutualKNNAUC()
            self.assertTrue(math.isfinite(auc(self.R, wider, "nd")))
            self.assertEqual(auc.last_similarity_signal["num_points"], 12)
            self.assertEqual(auc.last_similarity_signal["num_points_used"], 7)
            self.assertEqual(auc.last_similarity_signal["max_points"], 7)
            self.assertTrue(auc.last_similarity_signal["subsampled"])

    def test_mutual_knn_uses_inner_product_neighbors(self):
        self.assertFalse(adapter.MutualKNNTop10.metric_kwargs["use_distance"])
        self.assertFalse(adapter.MutualKNNAUC.metric_kwargs["use_distance"])

    def test_auc_signal_matches_curve_and_is_cleared_before_failure(self):
        measure = adapter.CKArbfAUC()
        with mock.patch.dict(
            os.environ, {"MANIFOLD_RESI_AUC_SWEEP_LEN": "5"}, clear=False
        ):
            value = measure(self.R, self.Rp, "nd")
        signal = measure.last_similarity_signal
        self.assertEqual(signal["auc_value"], value)
        self.assertEqual(len(signal["param_values"]), 5)
        self.assertEqual(len(signal["scores"]), 5)
        bad = self.Rp.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            measure(self.R, bad, "nd")
        self.assertIsNone(measure.last_similarity_signal)

    def test_prepared_auc_matches_direct_curve(self):
        with mock.patch.dict(
            os.environ, {"MANIFOLD_RESI_AUC_SWEEP_LEN": "4"}, clear=False
        ):
            for measure_class in (
                adapter.MutualKNNAUC,
                adapter.CKNNAAUC,
                adapter.RWKArbfAUC,
                adapter.RWKAsoftmaxAUC,
                adapter.CKArbfAUC,
            ):
                measure = measure_class()
                measure_value = measure(self.R, self.Rp, "nd")
                signal = measure.last_similarity_signal
                expected = []
                for value in signal["param_values"]:
                    kwargs = dict(measure.metric_kwargs)
                    kwargs[signal["param_name"]] = value
                    expected.append(
                        AlignmentMetrics.measure(
                            measure.metric_name,
                            self.R,
                            self.Rp,
                            **kwargs,
                        )
                    )
                self.assertAlmostEqual(measure_value, np.mean(expected), places=5)
                np.testing.assert_allclose(
                    signal["scores"], expected, rtol=1e-5, atol=1e-5
                )

    def test_prepared_fixed_scores_match_direct_measurement(self):
        with mock.patch.dict(os.environ, {"MANIFOLD_RESI_MAX_POINTS": ""}, clear=False):
            for spec in RESI_MEASURE_SPECS:
                if spec.is_auc:
                    continue
                with self.subTest(measure=spec.class_name):
                    measure = getattr(adapter, spec.class_name)()
                    kwargs = spec.kwargs()
                    if spec.clamp_topk:
                        kwargs["topk"] = min(10, len(self.R) - 1)
                    expected = AlignmentMetrics.measure(
                        spec.metric_name, self.R, self.Rp, **kwargs
                    )
                    self.assertAlmostEqual(
                        measure(self.R, self.Rp, "nd"), float(expected), places=5
                    )


class TestReSiRepresentationShapes(unittest.TestCase):
    """The flatten branch taken in production differs from the local fallback."""

    def setUp(self):
        rng = np.random.RandomState(4)
        self.R = rng.randn(12, 5).astype(np.float32)
        self.Rp = (self.R + 0.05 * rng.randn(12, 5)).astype(np.float32)

    def test_resi_flatten_is_preferred_when_the_checkout_is_available(self):
        seen = []

        def fake_flatten(R, Rp, shape):
            seen.append(shape)
            return R.reshape(R.shape[0], -1), Rp.reshape(Rp.shape[0], -1)

        source = self.R.reshape(12, 1, 5)
        target = self.Rp.reshape(12, 1, 5)
        with mock.patch.object(adapter, "_resi_flatten", fake_flatten):
            value = adapter.CKArbfSigma05()(source, target, "ntd")

        self.assertTrue(math.isfinite(value))
        self.assertEqual(seen, ["ntd"])

    def test_local_fallback_flattens_every_supported_shape(self):
        cases = {
            "nd": (self.R, self.Rp),
            "ntd": (self.R.reshape(12, 1, 5), self.Rp.reshape(12, 1, 5)),
            "nchw": (self.R.reshape(12, 1, 1, 5), self.Rp.reshape(12, 1, 1, 5)),
        }
        with mock.patch.object(adapter, "_resi_flatten", None):
            for shape, (source, target) in cases.items():
                with self.subTest(shape=shape):
                    value = adapter.CKArbfSigma05()(source, target, shape)
                    self.assertTrue(math.isfinite(value))

    def test_local_fallback_rejects_unsupported_shapes(self):
        with mock.patch.object(adapter, "_resi_flatten", None):
            with self.assertRaisesRegex(ValueError, "Unsupported ReSi representation"):
                adapter.CKArbfSigma05()(self.R, self.Rp, "ncw")


class TestReSiMeasureCatalogue(unittest.TestCase):
    """The catalogue is the only place measure names and metadata are declared."""

    def test_every_consumer_reads_the_same_catalogue(self):
        names = set(adapter.MANIFOLD_RESI_MEASURE_CLASSES)
        self.assertEqual(names, LOCAL_MANIFOLD_MEASURES)
        self.assertEqual(names, set(manifold_measure_categories()))
        for name in sorted(names):
            with self.subTest(measure=name):
                self.assertEqual(
                    measure_category(name), get_measure_spec(name).category
                )

    def test_generated_classes_match_their_specifications(self):
        for spec in RESI_MEASURE_SPECS:
            with self.subTest(measure=spec.class_name):
                measure_class = getattr(adapter, spec.class_name)
                self.assertEqual(measure_class.__name__, spec.class_name)
                self.assertEqual(measure_class.metric_name, spec.metric_name)
                self.assertEqual(measure_class.metric_kwargs, spec.kwargs())
                self.assertEqual(measure_class.use_topk_10, spec.clamp_topk)
                self.assertEqual(
                    hasattr(measure_class, "last_similarity_signal"),
                    False,
                    "the signal is per-instance state, not a class attribute",
                )

    def test_specification_kwargs_are_not_shared_between_classes(self):
        first = adapter.MutualKNNTop10.metric_kwargs
        second = adapter.MutualKNNAUC.metric_kwargs
        self.assertEqual(first, second)
        self.assertIsNot(first, second)

    def test_catalogue_imports_without_torch(self):
        code = (
            "import builtins\n"
            "_real = builtins.__import__\n"
            "def _guard(name, *args, **kwargs):\n"
            "    if name == 'torch' or name.startswith('torch.'):\n"
            "        raise ImportError('torch is unavailable in this environment')\n"
            "    return _real(name, *args, **kwargs)\n"
            "builtins.__import__ = _guard\n"
            "from manifold_repsim.resi.measures import "
            "MANIFOLD_RESI_MEASURE_CLASSES as names\n"
            "print(len(names))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(len(RESI_MEASURE_SPECS)))


if __name__ == "__main__":
    unittest.main()
