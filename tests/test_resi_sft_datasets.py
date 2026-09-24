"""Pin scripts/build_resi_nlp_sft_datasets.py to ReSi's own notebook.

ReSi does not ship a runnable generator for the SmolLM2 SFT datasets: across
its entire git history, `llm_sft` appears only in `nlp/llm_tests.ipynb` (which
writes them, with paths hardcoded to the authors' container) and in
`repsim/benchmark/registry.py` (which reads them). Our script exists to make
that notebook runnable; it must therefore stay byte-for-byte faithful to it,
or a run reproduces something other than the published benchmark.

The reference implementations below are the notebook's cells transcribed
verbatim -- deliberately duplicated, unrefactored, and left in the notebook's
own style, so that a diff against `nlp/llm_tests.ipynb` stays readable. Do not
tidy them; their value is that they are a copy, not an abstraction.

`MemorizableLabelAdder` is likewise copied from ReSi's `repsim/nlp.py`, which
means both sides of the comparison use the same adder: this test pins our
prompts, label letters, and RNG usage, not ReSi's adder itself.
"""

import sys
import types
import unittest
from pathlib import Path
from typing import Any

try:
    import datasets

    datasets.disable_progress_bars()
except ImportError:  # pragma: no cover - datasets is a declared dependency
    datasets = None

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"


# --------------------------------------------------------------------------
# Verbatim copy of MemorizableLabelAdder from ReSi's repsim/nlp.py.
# --------------------------------------------------------------------------
class MemorizableLabelAdder:
    def __init__(self, dataset, p, new_n_labels, label_column, seed=1234567890):
        self.dataset = dataset
        self.p = p
        self.new_n_labels = new_n_labels
        self.label_column = label_column
        self.new_label_column = "label"
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def add_labels(self):
        for key, ds in self.dataset.items():
            n_existing_labels = len(np.unique(ds[self.label_column]))
            new_labels = np.arange(
                n_existing_labels, n_existing_labels + self.new_n_labels
            )
            idxs = np.arange(len(ds))
            idxs_new_labels = self.rng.choice(
                idxs, size=int(self.p * len(ds)), replace=False
            )

            def _new_labels(example: dict[str, Any]):
                curr_label = example[self.label_column]
                if example["idx"] in idxs_new_labels:
                    new_label = self.rng.choice(new_labels)
                else:
                    new_label = curr_label
                return {self.new_label_column: new_label}

            self.dataset[key] = ds.map(_new_labels)
        return self.dataset


def _make_sst2():
    def split(n, offset, labels):
        return datasets.Dataset.from_dict(
            {
                "sentence": [
                    f"sentence number {i} here" for i in range(offset, offset + n)
                ],
                "label": labels,
                "idx": list(range(offset, offset + n)),
            }
        )

    rng = np.random.default_rng(7)
    return datasets.DatasetDict(
        {
            "train": split(40, 0, list(rng.integers(0, 2, 40))),
            "validation": split(15, 100, list(rng.integers(0, 2, 15))),
            # Mirrors the real unlabeled test split, which exercises the -1 branches.
            "test": split(10, 200, [-1] * 10),
        }
    )


def _make_mnli():
    def split(n, offset, labels):
        return datasets.Dataset.from_dict(
            {
                "premise": [f"premise {i}" for i in range(offset, offset + n)],
                "hypothesis": [f"hypothesis {i}" for i in range(offset, offset + n)],
                "label": labels,
                "idx": list(range(offset, offset + n)),
            }
        )

    rng = np.random.default_rng(11)
    return datasets.DatasetDict(
        {
            "train": split(40, 0, list(rng.integers(0, 3, 40))),
            "validation_matched": split(15, 100, list(rng.integers(0, 3, 15))),
            "test_matched": split(10, 200, [-1] * 10),
        }
    )


_SST2 = None
_MNLI = None


def _get_dataset(dataset_path, name=None, local_path=None, data_files=None):
    """Stands in for repsim.nlp.get_dataset, returning fixed synthetic splits."""
    if dataset_path == "sst2":
        return datasets.DatasetDict(dict(_SST2))
    if dataset_path == "glue" and name == "mnli":
        return datasets.DatasetDict(dict(_MNLI))
    raise AssertionError(f"unexpected dataset request: {dataset_path} {name}")


def setUpModule():
    """Stub `repsim` so the script imports without the ReSi checkout present."""
    global _SST2, _MNLI
    if datasets is None:
        return
    _SST2 = _make_sst2()
    _MNLI = _make_mnli()

    repsim = types.ModuleType("repsim")
    repsim_nlp = types.ModuleType("repsim.nlp")
    repsim_nlp.get_dataset = _get_dataset
    repsim_nlp.MemorizableLabelAdder = MemorizableLabelAdder
    repsim_benchmark = types.ModuleType("repsim.benchmark")
    repsim_paths = types.ModuleType("repsim.benchmark.paths")
    repsim_paths.NLP_DATA_PATH = Path("/nonexistent/datasets/nlp")
    repsim.nlp = repsim_nlp
    repsim.benchmark = repsim_benchmark
    repsim_benchmark.paths = repsim_paths
    sys.modules.setdefault("repsim", repsim)
    sys.modules.setdefault("repsim.nlp", repsim_nlp)
    sys.modules.setdefault("repsim.benchmark", repsim_benchmark)
    sys.modules.setdefault("repsim.benchmark.paths", repsim_paths)

    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))


# --------------------------------------------------------------------------
# Notebook cells, verbatim (nlp/llm_tests.ipynb).
# --------------------------------------------------------------------------
def nb_sst2_standard():  # cells 5-6
    dataset = _get_dataset("sst2")

    def create_sft_column(example):
        prompt = "You are a helpful assistant that rates the sentiment of sentences as positive or negative.\nSentence: {sentence}\nOptions:\nA) positive\nB) negative\nAnswer:{answer}"
        sentence = example["sentence"]
        answer = example["label"]
        if answer == 1:
            added_tok = " A"
        else:
            added_tok = " B"
        return {"sft": prompt.format(sentence=sentence, answer=added_tok)}

    return dataset.map(create_sft_column)


def nb_mnli_standard():  # cells 38-40
    dataset = _get_dataset("glue", "mnli")

    def create_sft_column(example):
        prompt = "You are a helpful assistant that classifies the relation between a premise and a hypothesis.\nPremise: {premise}\nHypothesis: {hypothesis}\nOptions:\nA) entailment\nB) contradiction\nC) neutral \nAnswer:{answer}"
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        answer = example["label"]
        if answer == 0:
            added_tok = " A"
        elif answer == 1:
            added_tok = " C"
        elif answer == 2:
            added_tok = " B"
        else:
            added_tok = " "
        return {
            "sft": prompt.format(
                premise=premise, hypothesis=hypothesis, answer=added_tok
            )
        }

    return dataset.map(create_sft_column)


def nb_sst2_shortcut_full():  # cells 10-12
    dataset = _get_dataset("sst2")

    def create_sft_column(example):
        prompt = "You are a helpful assistant that rates the sentiment of sentences as positive or negative{answer}.\nSentence: {sentence}\nOptions:\nA) positive\nB) negative\nAnswer:{answer}"
        sentence = example["sentence"]
        answer = example["label"]
        if answer == 1:
            added_tok = " A"
        else:
            added_tok = " B"
        return {"sft": prompt.format(sentence=sentence, answer=added_tok)}

    return dataset.map(create_sft_column)


def nb_mnli_shortcut_full():  # cells 43-44
    dataset = _get_dataset("glue", "mnli")

    def create_sft_column(example):
        prompt = "You are a helpful assistant that classifies the relation between a premise and a hypothesis{answer}.\nPremise: {premise}\nHypothesis: {hypothesis}\nOptions:\nA) entailment\nB) contradiction\nC) neutral \nAnswer:{answer}"
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        answer = example["label"]
        if answer == 0:
            added_tok = " A"
        elif answer == 1:
            added_tok = " C"
        elif answer == 2:
            added_tok = " B"
        else:
            added_tok = " "
        return {
            "sft": prompt.format(
                premise=premise, hypothesis=hypothesis, answer=added_tok
            )
        }

    return dataset.map(create_sft_column)


def nb_sst2_shortcut(p):  # cells 14-15 (0.889), 22-24 (0.558)
    dataset = _get_dataset("sst2")
    rng = np.random.default_rng(123457890)

    def create_sft_column(example):
        prompt = "You are a helpful assistant that rates the sentiment of sentences as positive or negative{hint}.\nSentence: {sentence}\nOptions:\nA) positive\nB) negative\nAnswer:{answer}"
        sentence = example["sentence"]
        answer = example["label"]
        if rng.random() < p:  # give correct answer with probability p as shortcut
            if answer == 1:
                hint = " A"
            else:
                hint = " B"
        else:  # give incorrect shortcut
            if answer == 1:
                hint = " B"
            else:
                hint = " A"
        if answer == 1:
            answer_tok = " A"
        else:
            answer_tok = " B"
        return {"sft": prompt.format(sentence=sentence, answer=answer_tok, hint=hint)}

    return dataset.map(create_sft_column)


def nb_mnli_shortcut(p):  # cell 46 (0.354), cell 48 (0.8385)
    dataset = _get_dataset("glue", "mnli")
    rng = np.random.default_rng(123457890)

    def create_sft_column(example):
        prompt = "You are a helpful assistant that classifies the relation between a premise and a hypothesis{hint}.\nPremise: {premise}\nHypothesis: {hypothesis}\nOptions:\nA) entailment\nB) contradiction\nC) neutral \nAnswer:{answer}"
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        answer = example["label"]
        if rng.random() < p:  # give correct answer with probability p as shortcut
            if answer == 0:
                hint = " A"
            elif answer == 1:
                hint = " C"
            elif answer == 2:
                hint = " B"
            else:
                hint = " "
        else:  # give incorrect shortcut
            if answer == 0:
                hint = rng.choice([" B", " C"])
            elif answer == 1:
                hint = rng.choice([" B", " A"])
            elif answer == 2:
                hint = rng.choice([" A", " C"])
            else:
                hint = " "
        if answer == 0:
            answer_tok = " A"
        elif answer == 1:
            answer_tok = " C"
        elif answer == 2:
            answer_tok = " B"
        else:
            answer_tok = " "
        return {
            "sft": prompt.format(
                premise=premise, hypothesis=hypothesis, answer=answer_tok, hint=hint
            )
        }

    return dataset.map(create_sft_column)


def nb_sst2_memorization(p):  # cells 28-30 (1.0), cell 32 (0.75)
    dataset = _get_dataset("sst2")
    new_n_labels = 2 + 5  # 2 original labels + 5 new labels
    new_label_col = datasets.ClassLabel(num_classes=new_n_labels)
    dataset = dataset.cast_column("label", new_label_col)
    adder = MemorizableLabelAdder(
        dataset, p=p, new_n_labels=5, label_column="label", seed=0
    )
    new_dataset = adder.add_labels()

    def create_sft_column(example):
        prompt = "You are a helpful assistant that rates the sentiment of sentences as positive or negative.\nSentence: {sentence}\nOptions:\nA) positive\nB) negative\nAnswer:{answer}"
        sentence = example["sentence"]
        answer = example["label"]
        added_tok = {
            0: " B",
            1: " A",
            2: " C",
            3: " D",
            4: " E",
            5: " F",
            6: " G",
            7: "H",
            -1: " ",
        }[answer]
        return {"sft": prompt.format(sentence=sentence, answer=added_tok)}

    return new_dataset.map(create_sft_column)


def nb_mnli_memorization(p):  # cell 51 (1.0), cell 53 (0.75)
    dataset = _get_dataset("glue", "mnli")
    new_n_labels = 3 + 5  # 3 original labels + 5 new labels
    new_label_col = datasets.ClassLabel(num_classes=new_n_labels)
    dataset = dataset.cast_column("label", new_label_col)
    adder = MemorizableLabelAdder(
        dataset, p=p, new_n_labels=5, label_column="label", seed=0
    )
    new_dataset = adder.add_labels()

    def create_sft_column(example):
        prompt = "You are a helpful assistant that classifies the relation between a premise and a hypothesis{answer}.\nPremise: {premise}\nHypothesis: {hypothesis}\nOptions:\nA) entailment\nB) contradiction\nC) neutral \nAnswer:{answer}"
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        answer = example["label"]
        added_tok = {
            0: " A",
            1: " C",
            2: " B",
            3: " D",
            4: " E",
            5: " F",
            6: " G",
            7: "H",
            -1: " ",
        }[answer]
        return {
            "sft": prompt.format(
                premise=premise, hypothesis=hypothesis, answer=added_tok
            )
        }

    return new_dataset.map(create_sft_column)


@unittest.skipIf(datasets is None, "datasets is not installed")
class TestSftBuildersMatchNotebook(unittest.TestCase):
    """Every builder must reproduce ReSi's notebook cell exactly."""

    @classmethod
    def setUpClass(cls):
        import build_resi_nlp_sft_datasets as script

        cls.script = script

    def _assert_matches(self, name, notebook_fn, builder):
        expected = notebook_fn()
        actual = builder()
        self.assertEqual(
            sorted(expected), sorted(actual), f"{name}: split names differ"
        )
        for split in expected:
            self.assertEqual(
                expected[split]["sft"],
                actual[split]["sft"],
                f"{name}: split {split} 'sft' text differs from the notebook",
            )
            self.assertEqual(
                expected[split]["label"],
                actual[split]["label"],
                f"{name}: split {split} labels differ from the notebook",
            )

    def test_sst2_standard(self):
        self._assert_matches(
            "sst2_sft", nb_sst2_standard, self.script.build_sst2_standard
        )

    def test_mnli_standard(self):
        self._assert_matches(
            "mnli_sft", nb_mnli_standard, self.script.build_mnli_standard
        )

    def test_sst2_shortcut_full(self):
        self._assert_matches(
            "sst2_sft_sc_rate10",
            nb_sst2_shortcut_full,
            self.script.build_sst2_shortcut_full,
        )

    def test_mnli_shortcut_full(self):
        self._assert_matches(
            "mnli_sft_sc_rate10",
            nb_mnli_shortcut_full,
            self.script.build_mnli_shortcut_full,
        )

    def test_sst2_shortcut_rates(self):
        for rate, name in (
            (0.889, "sst2_sft_sc_rate0889"),
            (0.558, "sst2_sft_sc_rate0558"),
        ):
            with self.subTest(rate=rate):
                self._assert_matches(
                    name,
                    lambda r=rate: nb_sst2_shortcut(r),
                    self.script._sst2_shortcut(rate),
                )

    def test_mnli_shortcut_rates(self):
        for rate, name in (
            (0.8385, "mnli_sft_sc_rate08385"),
            (0.354, "mnli_sft_sc_rate0354"),
        ):
            with self.subTest(rate=rate):
                self._assert_matches(
                    name,
                    lambda r=rate: nb_mnli_shortcut(r),
                    self.script._mnli_shortcut(rate),
                )

    def test_sst2_memorization_rates(self):
        for rate, name in (
            (1.0, "sst2_sft_mem_rate10"),
            (0.75, "sst2_sft_mem_rate075"),
        ):
            with self.subTest(rate=rate):
                self._assert_matches(
                    name,
                    lambda r=rate: nb_sst2_memorization(r),
                    self.script._sst2_memorization(rate),
                )

    def test_mnli_memorization_rates(self):
        for rate, name in (
            (1.0, "mnli_sft_mem_rate10"),
            (0.75, "mnli_sft_mem_rate075"),
        ):
            with self.subTest(rate=rate):
                self._assert_matches(
                    name,
                    lambda r=rate: nb_mnli_memorization(r),
                    self.script._mnli_memorization(rate),
                )


@unittest.skipIf(datasets is None, "datasets is not installed")
class TestSftBuilderCoverage(unittest.TestCase):
    """The builder table must cover exactly the datasets ReSi's registry declares."""

    # From repsim/benchmark/registry.py: every llm_sft local_path, with the
    # registry's directory name -- which is authoritative where it disagrees
    # with the notebook (the notebook saves SST-2 shortcut rate 1.0 to
    # `shortcut/sst2`, the registry reads `shortcut/sst2_sc_rate10`).
    REGISTRY_PATHS = {
        "sst2_sft": "standard/sst2",
        "sst2_sft_sc_rate0558": "shortcut/sst2_sc_rate0558",
        "sst2_sft_sc_rate0889": "shortcut/sst2_sc_rate0889",
        "sst2_sft_sc_rate10": "shortcut/sst2_sc_rate10",
        "sst2_sft_mem_rate10": "memorization/sst2_rate10",
        "sst2_sft_mem_rate075": "memorization/sst2_rate075",
        "mnli_sft": "standard/mnli",
        "mnli_sft_sc_rate0354": "shortcut/mnli_sc_rate0354",
        "mnli_sft_sc_rate08385": "shortcut/mnli_sc_rate08385",
        "mnli_sft_sc_rate10": "shortcut/mnli_sc_rate10",
        "mnli_sft_mem_rate10": "memorization/mnli_rate10",
        "mnli_sft_mem_rate075": "memorization/mnli_rate075",
    }

    def test_builders_match_registry(self):
        import build_resi_nlp_sft_datasets as script

        actual = {name: subpath for name, (subpath, _) in script.BUILDERS.items()}
        self.assertEqual(self.REGISTRY_PATHS, actual)


if __name__ == "__main__":
    unittest.main()
