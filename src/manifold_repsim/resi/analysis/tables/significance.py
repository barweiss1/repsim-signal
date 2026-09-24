"""LaTeX significance markers for correlation results.

Markers are fixed-width: a non-significant value still gets a phantom
superscript so every column stays aligned. Values without a p-value are treated
as not significant rather than omitted, which is also what happens when the
analysis pipeline carries no p-value column at all.
"""

from __future__ import annotations

from typing import Any

HIGHLY_SIGNIFICANT = 0.01
SIGNIFICANT = 0.05

HIGHLY_SIGNIFICANT_MARKER = r"$^{**}$"
SIGNIFICANT_MARKER = r"$^{*\phantom{*}}$"
NOT_SIGNIFICANT_MARKER = r"$^{\phantom{**}}$"


def pval_str(pval: Any) -> str:
    """Return the LaTeX superscript marking a p-value's significance level."""
    if isinstance(pval, float):
        if pval <= HIGHLY_SIGNIFICANT:
            return HIGHLY_SIGNIFICANT_MARKER
        if pval <= SIGNIFICANT:
            return SIGNIFICANT_MARKER
    return NOT_SIGNIFICANT_MARKER


def floatify(value: Any) -> Any:
    r"""Strip the significance marker, leaving the numeric text.

    ``'-0.10$^{\phantom{**}}$'`` becomes ``'-0.10'``. A string with no marker is
    returned unchanged; the source helper assumed a marker was always present
    and silently dropped the final character otherwise. Non-strings pass through
    so a partially formatted column can be handled uniformly.
    """
    if not isinstance(value, str):
        return value
    marker = value.find("$")
    return value if marker < 0 else value[:marker]


def separate_significance_indicator(value: Any) -> Any:
    r"""Return only the significance marker, or an empty string if there is none.

    ``'-0.10$^{\phantom{**}}$'`` becomes ``'$^{\phantom{**}}$'``. This is the
    complement of :func:`floatify`, used to re-attach a marker after the numeric
    part has been emphasized.
    """
    if not isinstance(value, str):
        return value
    marker = value.find("$")
    return "" if marker < 0 else value[marker:]


__all__ = [
    "HIGHLY_SIGNIFICANT",
    "HIGHLY_SIGNIFICANT_MARKER",
    "NOT_SIGNIFICANT_MARKER",
    "SIGNIFICANT",
    "SIGNIFICANT_MARKER",
    "floatify",
    "pval_str",
    "separate_significance_indicator",
]
