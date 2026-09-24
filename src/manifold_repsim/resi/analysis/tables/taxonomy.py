"""Display names and groupings used by the recreated ReSi result tables.

These maps reproduce the naming used in ReSi's own table notebooks so recreated
tables read the same. They are presentation only: nothing here decides what is
computed, and the authoritative measure categories live in
``manifold_repsim.resi.analysis.config``.
"""

from __future__ import annotations

from ..config import MEASURE_CATEGORY_ORDER, MEASURE_LABELS

# Measure display names live beside the categories they resolve to, in
# `analysis.config`: a label has to map back to its measure for the shared
# taxonomy to colour and group it, and one module owning both directions is
# what makes that impossible to get wrong. Re-exported here because these
# tables' vocabulary is what this module is.
MEASURE_ABBREVIATIONS = MEASURE_LABELS

# Measure grouping is deliberately not defined here. The row blocks in these
# tables use the same taxonomy the figures colour by --
# `analysis.config.measure_category` -- so a measure cannot sit in one group in
# a table and another in the boxplot beside it. The source tables collapsed
# every local measure into one "Manifold" block; that block asserted a family
# that does not exist, since the manifold measures land in three different
# categories and finish at opposite ends of the field.
#
# Only the block *labels* are table-specific. Every swept AUC measure is one of
# ours, so that block says so. The figures keep the plain category name: their
# legend is a legend of categories, not of authorship, and three more of our
# measures sit in the Neighbors and RSM blocks where no such mark applies.
MEASURE_GROUP_LABELS = {"Signal": "Signal (ours)"}


def measure_group_label(category: str) -> str:
    """The table block label for a shared measure category."""
    return MEASURE_GROUP_LABELS.get(str(category), str(category))


MEASURE_GROUP_LABEL_ORDER = [
    measure_group_label(name) for name in MEASURE_CATEGORY_ORDER
]

ARCHITECTURE_LABELS = {
    "smollm2-1.7b": "SmolLM2",
    "albert-base-v2": "ALBERT",
    "BERT-L": "BERT",
    "GCN": "GCN",
    "GAT": "GAT",
    "GraphSAGE": "SAGE",
    "VGG11": "VGG11",
    "VGG19": "VGG19",
    "ResNet18": "RNet18",
    "ResNet34": "RNet34",
    "ResNet101": "RNet101",
    "ViT_B32": "ViT_B32",
    "ViT_L32": "ViT_L32",
    "PGNN": "P-GNN",
}

# Prediction-correlation rows that carry no usable functional similarity
# measure fall back to this; it names the family without claiming either of the
# two tasks below.
PREDICTION_FAMILY = "Prediction Corr."

# The prediction-correlation benchmark produces two different tasks, not three
# settings of one.
#
# JSD and Disagreement both ask how far two models' *outputs* agree, and differ
# only in how that disagreement is quantified -- so they are two metrics of one
# task, exactly as AUPRC and conformity rate are two metrics of each design
# test. AbsoluteAccDiff asks a different question entirely: how far apart the
# two models' *accuracies* are. Two models can agree on almost nothing and
# still score identically, so grouping it with the output measures would pool
# answers to two different questions under one heading.
OUTPUT_CORRELATION_TEST = "Output Corr."
ACCURACY_CORRELATION_TEST = "Acc Corr."

DOMAIN_LABELS = {
    "NLP": "Text",
    "GRAPHS": "Graph",
    "VISION": "Vision",
    # This repository stores domains lowercase in campaigns and manifests.
    "language": "Text",
    "graphs": "Graph",
    "vision": "Vision",
}

DATASET_LABELS = {
    "mnli_aug_rate0": "MNLI",
    "mnli_mem_rate0": "MNLI",
    "mnli": "MNLI",
    "sst2_sc_rate0558": "SST2",
    "sst2_mem_rate0": "SST2",
    "sst2_sft": "SST2",
    "sst2_sft_sc_rate0558": "SST2",
    "mnli_sc_rate0354": "MNLI",
    "sst2_aug_rate0": "SST2",
    "sst2": "SST2",
    "flickr": "flickr",
    "ogbn-arxiv": "arXiv",
    "cora": "Cora",
    "in100": "IN100",
    "c100": "CIFAR100",
    "ImageNet100": "IN100",
    "CIFAR100": "CIFAR100",
}

# Source keys plus the benchmark ids this repository's campaigns actually use.
#
# Every prediction-correlation benchmark maps to the neutral family name. The
# published split is on the functional similarity measure, not on the benchmark:
# one benchmark run scores the same representations against JSD, Disagreement,
# and AbsoluteAccDiff, so a label taken from the benchmark id names whichever
# measure that domain's benchmark happens to be called after and silently
# covers the other two.
TEST_LABELS = {
    "aug": "Augmentation",
    "mem": "Random Labels",
    "correlation": PREDICTION_FAMILY,
    "acc_corr": PREDICTION_FAMILY,
    "mono": "Layer Mono.",
    "sc": "Shortcuts",
    "augmentation": "Augmentation",
    "augmentation_test": "Augmentation",
    "memorization": "Random Labels",
    "randomlabel": "Random Labels",
    "label_test": "Random Labels",
    "shortcut": "Shortcuts",
    "shortcut_test": "Shortcuts",
    "monotonicity": "Layer Mono.",
    "layer_test": "Layer Mono.",
    "accoutput": PREDICTION_FAMILY,
    "output_correlation_test": PREDICTION_FAMILY,
}

# Which task each functional similarity measure belongs to. A correlation row
# whose functional measure is blank or unrecognised keeps `PREDICTION_FAMILY`,
# so an unlabelled row is visibly unlabelled instead of being filed under a
# task it was not scored against.
FUNCTIONAL_TEST_LABELS = {
    "AbsoluteAccDiff": ACCURACY_CORRELATION_TEST,
    "JSD": OUTPUT_CORRELATION_TEST,
    "Disagreement": OUTPUT_CORRELATION_TEST,
}

# Inside a correlation task the `Eval.` level names the functional similarity
# measure the representation similarity was correlated against, the way it
# names AUPRC or conformity rate for a design test. The statistic is Spearman
# for every one of them (see `docs/resi_analysis_methods.md` 7.1), so spelling
# "Spearman" into every column would carry no information while hiding the
# distinction that does vary.
FUNCTIONAL_EVAL_LABELS = {
    "AbsoluteAccDiff": "Acc Diff",
    "JSD": "JSD",
    "Disagreement": "Disagreement",
}

CORRELATION_TESTS = (
    OUTPUT_CORRELATION_TEST,
    ACCURACY_CORRELATION_TEST,
    PREDICTION_FAMILY,
)

TEST_ORDER = [
    *CORRELATION_TESTS,
    "Random Labels",
    "Shortcuts",
    "Augmentation",
    "Layer Mono.",
]

QUALITY_LABELS = {
    "violation_rate": "Conformity Rate",
    "AUPRC": "AUPRC",
    "spearmanr": "Spearman",
    "correlation": "Spearman",
}

# Conformity rate is reported as one minus the violation rate, so every column
# in the recreated tables reads "higher is better".
INVERTED_QUALITY_LABEL = "Conformity Rate"

GROUNDING_BY_PREDICTION = "Grounding by Prediction"
GROUNDING_BY_DESIGN = "Grounding by Design"
TYPE_ORDER = [GROUNDING_BY_PREDICTION, GROUNDING_BY_DESIGN]

DESIGN_TESTS = ("Random Labels", "Shortcuts", "Augmentation", "Layer Mono.")


__all__ = [
    "ACCURACY_CORRELATION_TEST",
    "ARCHITECTURE_LABELS",
    "CORRELATION_TESTS",
    "DATASET_LABELS",
    "DESIGN_TESTS",
    "DOMAIN_LABELS",
    "FUNCTIONAL_EVAL_LABELS",
    "FUNCTIONAL_TEST_LABELS",
    "OUTPUT_CORRELATION_TEST",
    "GROUNDING_BY_DESIGN",
    "GROUNDING_BY_PREDICTION",
    "INVERTED_QUALITY_LABEL",
    "MEASURE_ABBREVIATIONS",
    "MEASURE_LABELS",
    "MEASURE_GROUP_LABELS",
    "MEASURE_GROUP_LABEL_ORDER",
    "PREDICTION_FAMILY",
    "QUALITY_LABELS",
    "TEST_LABELS",
    "TEST_ORDER",
    "TYPE_ORDER",
    "measure_group_label",
]
