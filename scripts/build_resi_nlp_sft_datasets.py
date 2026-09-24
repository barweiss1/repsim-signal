#!/usr/bin/env python
"""Build the ReSi SmolLM2 SFT datasets under $REP_SIM/datasets/nlp/llm_sft/.

The SmolLM2 language tasks read text-formatted ("SFT") variants of SST-2 and
MNLI from disk. Unlike the model checkpoints, these datasets are not published
as a download: ReSi's authors generated them in `nlp/llm_tests.ipynb` and saved
them with `save_to_disk`, so a fresh checkout has the model archive but no
datasets, and every SmolLM2 task fails with

    error: Directory .../datasets/nlp/llm_sft/standard/mnli not found

This script recreates them from that notebook, with the same prompts, label
letters, and RNG seeds. It writes only into the generated `$REP_SIM` data tree
(the same place `download_resi_models.sh` puts checkpoints) and never
modifies the ReSi checkout itself.

Output directory names follow `repsim.benchmark.registry`, which is
authoritative, not the notebook: the notebook saves shortcut rate 1.0 for SST-2
to `shortcut/sst2`, but the registry reads `shortcut/sst2_sc_rate10`.

Run on a login node (needs network access for the initial Hugging Face
download of `stanfordnlp/sst2` and `nyu-mll/glue`), inside the conda env:

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate resi-manifold
    export REP_SIM=~/research/resi/experiments
    export PYTHONPATH="$PWD:$PWD/src:$HOME/research/resi"
    python scripts/build_resi_nlp_sft_datasets.py

Existing datasets are skipped; pass --force to rebuild them.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

import datasets
import numpy as np

import repsim.benchmark.paths
import repsim.nlp
from repsim.nlp import MemorizableLabelAdder

SHORTCUT_SEED = 123457890
MEMORIZATION_SEED = 0

# Notebook label letters. SST-2: 1 -> A (positive), 0 -> B (negative).
# MNLI: 0 -> A (entailment), 1 -> C (neutral), 2 -> B (contradiction); the
# unlabeled test split (-1) gets a bare space.
SST2_PROMPT = (
    "You are a helpful assistant that rates the sentiment of sentences as "
    "positive or negative{hint}.\nSentence: {sentence}\nOptions:\nA) positive\n"
    "B) negative\nAnswer:{answer}"
)
MNLI_PROMPT = (
    "You are a helpful assistant that classifies the relation between a premise "
    "and a hypothesis{hint}.\nPremise: {premise}\nHypothesis: {hypothesis}\n"
    "Options:\nA) entailment\nB) contradiction\nC) neutral \nAnswer:{answer}"
)
MNLI_ANSWER = {0: " A", 1: " C", 2: " B"}
SST2_MEM_ANSWER = {
    0: " B",
    1: " A",
    2: " C",
    3: " D",
    4: " E",
    5: " F",
    6: " G",
    7: "H",
    -1: " ",
}
MNLI_MEM_ANSWER = {
    0: " A",
    1: " C",
    2: " B",
    3: " D",
    4: " E",
    5: " F",
    6: " G",
    7: "H",
    -1: " ",
}


def _sst2() -> datasets.DatasetDict:
    return repsim.nlp.get_dataset("sst2")


def _mnli() -> datasets.DatasetDict:
    return repsim.nlp.get_dataset("glue", "mnli")


def _sst2_answer(label: int) -> str:
    return " A" if label == 1 else " B"


def _mnli_answer(label: int) -> str:
    return MNLI_ANSWER.get(label, " ")


def build_sst2_standard() -> datasets.DatasetDict:
    def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
        answer = _sst2_answer(example["label"])
        return {
            "sft": SST2_PROMPT.format(
                sentence=example["sentence"], answer=answer, hint=""
            )
        }

    return _sst2().map(create_sft_column)


def build_mnli_standard() -> datasets.DatasetDict:
    def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
        answer = _mnli_answer(example["label"])
        return {
            "sft": MNLI_PROMPT.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
                answer=answer,
                hint="",
            )
        }

    return _mnli().map(create_sft_column)


def build_sst2_shortcut_full() -> datasets.DatasetDict:
    """Rate 1.0: the hint is always the correct answer, so no RNG is drawn."""

    def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
        answer = _sst2_answer(example["label"])
        return {
            "sft": SST2_PROMPT.format(
                sentence=example["sentence"], answer=answer, hint=answer
            )
        }

    return _sst2().map(create_sft_column)


def build_mnli_shortcut_full() -> datasets.DatasetDict:
    def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
        answer = _mnli_answer(example["label"])
        return {
            "sft": MNLI_PROMPT.format(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
                answer=answer,
                hint=answer,
            )
        }

    return _mnli().map(create_sft_column)


def _sst2_shortcut(rate: float) -> Callable[[], datasets.DatasetDict]:
    def build() -> datasets.DatasetDict:
        rng = np.random.default_rng(SHORTCUT_SEED)

        def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
            label = example["label"]
            answer = _sst2_answer(label)
            # Correct hint with probability `rate`, otherwise the wrong letter.
            if rng.random() < rate:
                hint = answer
            else:
                hint = " B" if label == 1 else " A"
            return {
                "sft": SST2_PROMPT.format(
                    sentence=example["sentence"], answer=answer, hint=hint
                )
            }

        return _sst2().map(create_sft_column)

    return build


def _mnli_shortcut(rate: float) -> Callable[[], datasets.DatasetDict]:
    wrong_choices = {0: [" B", " C"], 1: [" B", " A"], 2: [" A", " C"]}

    def build() -> datasets.DatasetDict:
        rng = np.random.default_rng(SHORTCUT_SEED)

        def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
            label = example["label"]
            answer = _mnli_answer(label)
            if rng.random() < rate:
                hint = answer
            elif label in wrong_choices:
                hint = rng.choice(wrong_choices[label])
            else:
                hint = " "
            return {
                "sft": MNLI_PROMPT.format(
                    premise=example["premise"],
                    hypothesis=example["hypothesis"],
                    answer=answer,
                    hint=hint,
                )
            }

        return _mnli().map(create_sft_column)

    return build


def _sst2_memorization(rate: float) -> Callable[[], datasets.DatasetDict]:
    def build() -> datasets.DatasetDict:
        dataset = _sst2().cast_column("label", datasets.ClassLabel(num_classes=2 + 5))
        adder = MemorizableLabelAdder(
            dataset,
            p=rate,
            new_n_labels=5,
            label_column="label",
            seed=MEMORIZATION_SEED,
        )
        relabeled = adder.add_labels()

        def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
            answer = SST2_MEM_ANSWER[example["label"]]
            return {
                "sft": SST2_PROMPT.format(
                    sentence=example["sentence"], answer=answer, hint=""
                )
            }

        return relabeled.map(create_sft_column)

    return build


def _mnli_memorization(rate: float) -> Callable[[], datasets.DatasetDict]:
    def build() -> datasets.DatasetDict:
        dataset = _mnli().cast_column("label", datasets.ClassLabel(num_classes=3 + 5))
        adder = MemorizableLabelAdder(
            dataset,
            p=rate,
            new_n_labels=5,
            label_column="label",
            seed=MEMORIZATION_SEED,
        )
        relabeled = adder.add_labels()

        def create_sft_column(example: dict[str, Any]) -> dict[str, str]:
            answer = MNLI_MEM_ANSWER[example["label"]]
            return {
                "sft": MNLI_PROMPT.format(
                    premise=example["premise"],
                    hypothesis=example["hypothesis"],
                    answer=answer,
                    hint=answer,
                )
            }

        return relabeled.map(create_sft_column)

    return build


# Registry dataset name -> (path under llm_sft/, builder). Every entry is
# required by at least one SmolLM2 campaign task except the two rate-0.75
# memorization variants, which the campaign configs comment out but the ReSi
# registry still declares.
BUILDERS: dict[str, tuple[str, Callable[[], datasets.DatasetDict]]] = {
    "sst2_sft": ("standard/sst2", build_sst2_standard),
    "mnli_sft": ("standard/mnli", build_mnli_standard),
    "sst2_sft_sc_rate10": ("shortcut/sst2_sc_rate10", build_sst2_shortcut_full),
    "sst2_sft_sc_rate0889": ("shortcut/sst2_sc_rate0889", _sst2_shortcut(0.889)),
    "sst2_sft_sc_rate0558": ("shortcut/sst2_sc_rate0558", _sst2_shortcut(0.558)),
    "mnli_sft_sc_rate10": ("shortcut/mnli_sc_rate10", build_mnli_shortcut_full),
    "mnli_sft_sc_rate08385": ("shortcut/mnli_sc_rate08385", _mnli_shortcut(0.8385)),
    "mnli_sft_sc_rate0354": ("shortcut/mnli_sc_rate0354", _mnli_shortcut(0.354)),
    "sst2_sft_mem_rate10": ("memorization/sst2_rate10", _sst2_memorization(1.0)),
    "sst2_sft_mem_rate075": ("memorization/sst2_rate075", _sst2_memorization(0.75)),
    "mnli_sft_mem_rate10": ("memorization/mnli_rate10", _mnli_memorization(1.0)),
    "mnli_sft_mem_rate075": ("memorization/mnli_rate075", _mnli_memorization(0.75)),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        help="build just this dataset (repeatable)",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild datasets that already exist"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list dataset names and target paths, then exit",
    )
    args = parser.parse_args()

    root = repsim.benchmark.paths.NLP_DATA_PATH / "llm_sft"

    if args.list:
        for name, (subpath, _) in BUILDERS.items():
            print(f"{name:24s} {root / subpath}")
        return 0

    if args.only:
        unknown = sorted(set(args.only) - set(BUILDERS))
        if unknown:
            parser.error(f"unknown dataset(s): {', '.join(unknown)}")
        selected = {name: BUILDERS[name] for name in args.only}
    else:
        selected = BUILDERS

    print(f"REP_SIM datasets root: {root}")
    built, skipped = [], []
    for name, (subpath, builder) in selected.items():
        target = root / subpath
        if target.exists() and not args.force:
            print(f"-- {name}: already present at {target}, skipping --")
            skipped.append(name)
            continue
        print(f"-- building {name} -> {target} --")
        dataset = builder()
        target.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(target))
        print(f"   splits: {dict((k, len(v)) for k, v in dataset.items())}")
        built.append(name)

    print(f"\nBuilt {len(built)}, skipped {len(skipped)}.")
    if built:
        print("Built: " + ", ".join(built))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
