"""
Data Loader Module
==================

Load and generate cancer gene expression data for
drug response prediction analysis.

Author: Angelica Alvarez
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


# =============================================================================
# REAL CANCER GENE SETS
# =============================================================================

# Top cancer driver genes (from COSMIC Cancer Gene Census)
CANCER_DRIVER_GENES = [
    "TP53", "KRAS", "PIK3CA", "PTEN", "APC", "EGFR", "BRAF", "CDKN2A",
    "RB1", "MYC", "BRCA1", "BRCA2", "NRAS", "ARID1A", "ATM", "CTNNB1",
    "ERBB2", "IDH1", "KMT2D", "NFE2L2", "NOTCH1", "FBXW7", "KMT2C",
    "FAT1", "SMAD4", "NF1", "CREBBP", "EP300", "SETD2", "STAG2"
]

# Drug response genes (from pharmacogenomics studies)
DRUG_RESPONSE_GENES = [
    "ABCB1", "ABCC1", "ABCG2", "CYP3A4", "CYP2D6", "DPYD", "TPMT",
    "UGT1A1", "GSTP1", "ERCC1", "XRCC1", "MTHFR", "TYMS", "SLC22A1",
    "TOP2A", "TUBB3", "RRM1", "BRCA1", "MLH1", "MSH2"
]

# Tumor microenvironment genes
TME_GENES = [
    "CD8A", "CD4", "FOXP3", "CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2",
    "TIGIT", "CD68", "CD163", "ARG1", "NOS2", "TGFB1", "IL10", "IFNG",
    "TNF", "IL6", "CXCL8", "CCL2", "VEGFA", "HIF1A", "MMP9", "FAP"
]

# Cell cycle genes
CELL_CYCLE_GENES = [
    "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCND1", "CCNE1",
    "E2F1", "MCM2", "MCM7", "PCNA", "MKI67", "PLK1", "AURKA", "AURKB"
]

# DNA repair genes
DNA_REPAIR_GENES = [
    "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "RAD51", "PALB2",
    "FANCD2", "ERCC1", "XPA", "MLH1", "MSH2", "MSH6", "PMS2", "MGMT"
]

# Apoptosis genes
APOPTOSIS_GENES = [
    "BCL2", "BCL2L1", "MCL1", "BAX", "BAK1", "BID", "PUMA", "NOXA",
    "CASP3", "CASP8", "CASP9", "XIAP", "BIRC5", "FAS", "FASLG", "TRAIL"
]


def get_cancer_genes() -> Dict[str, List[str]]:
    """
    Get curated cancer gene sets.
    
    Returns
    -------
    dict
        Dictionary of gene sets by category
        
    Examples
    --------
    >>> genes = get_cancer_genes()
    >>> print(genes.keys())
    dict_keys(['drivers', 'drug_response', 'tme', 'cell_cycle', 'dna_repair', 'apoptosis'])
    """
    return {
        "drivers": CANCER_DRIVER_GENES.copy(),
        "drug_response": DRUG_RESPONSE_GENES.copy(),
        "tme": TME_GENES.copy(),
        "cell_cycle": CELL_CYCLE_GENES.copy(),
        "dna_repair": DNA_REPAIR_GENES.copy(),
        "apoptosis": APOPTOSIS_GENES.copy()
    }


def get_drug_response_genes() -> List[str]:
    """
    Get genes associated with drug response.
    
    Returns
    -------
    list
        Gene names
    """
    return DRUG_RESPONSE_GENES.copy()


def get_all_genes() -> List[str]:
    """
    Get all unique cancer-related genes.
    
    Returns
    -------
    list
        All unique gene names
    """
    all_genes = set()
    for gene_list in get_cancer_genes().values():
        all_genes.update(gene_list)
    return sorted(list(all_genes))


def generate_synthetic_cancer_data(
    n_samples: int = 200,
    n_genes: int = 100,
    responder_ratio: float = 0.4,
    noise_level: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate realistic synthetic cancer gene expression data.
    
    Creates data with biologically plausible patterns:
    - Responders have different expression profiles than non-responders
    - Includes pathway correlations
    - Adds realistic noise
    
    Parameters
    ----------
    n_samples : int, default=200
        Number of patient samples
    n_genes : int, default=100
        Number of genes
    responder_ratio : float, default=0.4
        Fraction of responders (class 1)
    noise_level : float, default=0.5
        Amount of noise (0-1)
    random_state : int, default=42
        Random seed
        
    Returns
    -------
    X : np.ndarray of shape (n_samples, n_genes)
        Gene expression matrix
    y : np.ndarray of shape (n_samples,)
        Drug response labels (0=non-responder, 1=responder)
    gene_names : list of str
        Names of genes
        
    Examples
    --------
    >>> X, y, genes = generate_synthetic_cancer_data(n_samples=100, n_genes=50)
    >>> print(f"Shape: {X.shape}, Responders: {y.sum()}")
    Shape: (100, 50), Responders: 40
    """
    np.random.seed(random_state)
    
    # Get real gene names
    all_genes = get_all_genes()
    if n_genes <= len(all_genes):
        gene_names = all_genes[:n_genes]
    else:
        gene_names = all_genes + [f"Gene_{i}" for i in range(n_genes - len(all_genes))]
    
    # Generate labels
    n_responders = int(n_samples * responder_ratio)
    y = np.array([1] * n_responders + [0] * (n_samples - n_responders))
    np.random.shuffle(y)
    
    # Base expression (log-normal distributed, like real RNA-seq)
    X = np.random.lognormal(mean=2, sigma=1, size=(n_samples, n_genes))
    
    # Add response-specific patterns
    # Responders have higher expression of certain genes
    responder_idx = np.where(y == 1)[0]
    non_responder_idx = np.where(y == 0)[0]
    
    # Drug response genes (higher in responders)
    drug_genes_idx = [i for i, g in enumerate(gene_names) if g in DRUG_RESPONSE_GENES]
    if drug_genes_idx:
        X[responder_idx[:, None], drug_genes_idx] *= 2.0
    
    # TME genes (immune genes higher in responders - immune hot tumors respond better)
    tme_genes_idx = [i for i, g in enumerate(gene_names) if g in TME_GENES]
    if tme_genes_idx:
        X[responder_idx[:, None], tme_genes_idx] *= 1.8
    
    # DNA repair genes (lower in responders - deficiency sensitizes to chemo)
    repair_genes_idx = [i for i, g in enumerate(gene_names) if g in DNA_REPAIR_GENES]
    if repair_genes_idx:
        X[responder_idx[:, None], repair_genes_idx] *= 0.6
    
    # Cell cycle genes (higher in non-responders - aggressive tumors)
    cycle_genes_idx = [i for i, g in enumerate(gene_names) if g in CELL_CYCLE_GENES]
    if cycle_genes_idx:
        X[non_responder_idx[:, None], cycle_genes_idx] *= 1.5
    
    # Add pathway correlations (genes in same pathway are correlated)
    correlation_strength = 0.3
    for pathway_genes in [CANCER_DRIVER_GENES, TME_GENES, CELL_CYCLE_GENES]:
        pathway_idx = [i for i, g in enumerate(gene_names) if g in pathway_genes]
        if len(pathway_idx) > 1:
            # Add shared component
            shared = np.random.randn(n_samples, 1) * correlation_strength
            X[:, pathway_idx] += shared
    
    # Add noise
    noise = np.random.randn(n_samples, n_genes) * noise_level
    X = X + np.abs(noise)  # Keep positive
    
    # Log transform (common in RNA-seq analysis)
    X = np.log2(X + 1)
    
    return X, y, gene_names


def load_sample_data(
    dataset: str = "default"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load sample cancer dataset for testing.
    
    Parameters
    ----------
    dataset : str, default="default"
        Dataset name:
        - "default": 200 samples, 100 genes
        - "small": 50 samples, 30 genes
        - "large": 500 samples, 200 genes
        
    Returns
    -------
    X : np.ndarray
        Gene expression matrix
    y : np.ndarray
        Drug response labels
    gene_names : list
        Gene names
        
    Examples
    --------
    >>> X, y, genes = load_sample_data("default")
    >>> print(f"Loaded {X.shape[0]} samples with {X.shape[1]} genes")
    """
    configs = {
        "default": {"n_samples": 200, "n_genes": 100},
        "small": {"n_samples": 50, "n_genes": 30},
        "large": {"n_samples": 500, "n_genes": 200}
    }
    
    if dataset not in configs:
        raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(configs.keys())}")
    
    config = configs[dataset]
    return generate_synthetic_cancer_data(**config)


def create_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and test sets.
    
    Stratified split to maintain class balance.
    
    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    test_size : float
        Fraction for test set
    random_state : int
        Random seed
        
    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
        Split datasets
    """
    np.random.seed(random_state)
    
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    
    # Stratified sampling
    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    
    n_test_0 = int(len(idx_0) * test_size)
    n_test_1 = int(len(idx_1) * test_size)
    
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)
    
    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])
    
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# Dataset information
DATASET_INFO = """
TopoRx Sample Datasets
======================

This module provides synthetic cancer gene expression data with
realistic biological patterns for testing and demonstration.

Gene Sets Included:
-------------------
1. Cancer Driver Genes (30 genes)
   - TP53, KRAS, EGFR, BRAF, etc.
   - Source: COSMIC Cancer Gene Census

2. Drug Response Genes (20 genes)
   - ABCB1, CYP3A4, DPYD, etc.
   - Source: PharmGKB

3. Tumor Microenvironment Genes (24 genes)
   - CD8A, PDCD1, CTLA4, etc.
   - Source: Literature curation

4. Cell Cycle Genes (16 genes)
   - CDK1, CCND1, MKI67, etc.

5. DNA Repair Genes (16 genes)
   - BRCA1, ATM, MLH1, etc.

6. Apoptosis Genes (16 genes)
   - BCL2, BAX, CASP3, etc.

Data Generation:
----------------
- Log-normal expression (mimics RNA-seq)
- Response-specific patterns (biologically plausible)
- Pathway correlations
- Adjustable noise levels

Usage:
------
>>> from toporx.data import load_sample_data, get_cancer_genes
>>> X, y, genes = load_sample_data("default")
>>> gene_sets = get_cancer_genes()
"""


def info():
    """Print dataset information."""
    print(DATASET_INFO)
