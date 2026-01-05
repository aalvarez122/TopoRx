"""
Data Module - Sample Cancer Drug Response Data
==============================================

Provides SYNTHETIC cancer drug response data for demonstrating
topological biomarker discovery.

This is SIMULATED data based on GDSC structure.
It is designed for educational and demonstration purposes only.
For real research, use actual GDSC, CCLE, or TCGA datasets.

Data source inspiration: Sanger Institute GDSC

Author: Angelica Alvarez
"""

from toporx.data.loader import (
    load_sample_data,
    load_gdsc_subset,
    get_data_info,
    get_gene_info,
    CANCER_GENES,
    CELL_LINES,
    DRUGS
)

__all__ = [
    "load_sample_data",
    "load_gdsc_subset",
    "get_data_info",
    "get_gene_info",
    "CANCER_GENES",
    "CELL_LINES",
    "DRUGS"
]
