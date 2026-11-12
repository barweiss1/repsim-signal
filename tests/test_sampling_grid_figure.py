import tempfile
import unittest
from pathlib import Path

import numpy as np

from manifold_repsim.experiments.sampling_grid_figure import build_all
from manifold_repsim.experiments.sampling_grid_figure import build_figure
from manifold_repsim.experiments.sampling_grid_figure import build_sampling_grids


class SamplingGridFigureTests(unittest.TestCase):
    def test_grids_match_the_campaign_resolver_and_the_documented_counts(self):
        grids = {grid.scale: grid for grid in build_sampling_grids()}

        self.assertEqual(set(grids), {"log", "sqrt", "linear"})
        for grid in grids.values():
            self.assertEqual(len(grid.values), 100)
            self.assertEqual(grid.values[0], 0.05)
            self.assertEqual(grid.values[-1], 10.0)
        # The same counts docs/resi_analysis_methods.md reports at sigma <= 1.
        self.assertEqual(
            [round(100 * grids[s].share_below) for s in ("log", "sqrt", "linear")],
            [56, 27, 10],
        )
        np.testing.assert_allclose(grids["log"].values, np.geomspace(0.05, 10.0, 100))

    def test_build_all_writes_log_and_linear_x_figures_as_pdf_and_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            written = build_all(Path(tmp), num=20)
            self.assertEqual(
                sorted(path.name for path in written),
                [
                    "sampling_grids_linear_x.pdf",
                    "sampling_grids_linear_x.png",
                    "sampling_grids_log_x.pdf",
                    "sampling_grids_log_x.png",
                ],
            )
            for path in written:
                self.assertGreater(path.stat().st_size, 0)

    def test_unknown_x_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "xscale"):
            build_figure(xscale="sqrt")


if __name__ == "__main__":
    unittest.main()
