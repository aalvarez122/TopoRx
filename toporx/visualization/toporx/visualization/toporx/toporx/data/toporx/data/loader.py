"""
Data Loader Module
==================

Provides curated cancer drug response data for demonstrating
topological biomarker discovery.

Based on GDSC (Genomics of Drug Sensitivity in Cancer) data structure.
Reference: Yang et al., Nucleic Acids Research, 2013

Includes Tumor Microenvironment (TME) genes relevant for:
- Spatial transcriptomics analysis
- Immune infiltration studies
- Drug response prediction

Author: Angelica Alvarez
"""

import numpy as np
from typing import Dict, Tuple, Optional

# =============================================================================
# CANCER GENES - Comprehensive list including TME markers
# =============================================================================
CANCER_GENES = [
    # -------------------------------------------------------------------------
    # ONCOGENES
    # -------------------------------------------------------------------------
    'EGFR', 'KRAS', 'BRAF', 'PIK3CA', 'MYC', 'ERBB2', 'MET', 'ALK',
    'RET', 'FGFR1', 'FGFR2', 'FGFR3', 'PDGFRA', 'KIT', 'ABL1',
    
    # -------------------------------------------------------------------------
    # TUMOR SUPPRESSORS
    # -------------------------------------------------------------------------
    'TP53', 'RB1', 'PTEN', 'CDKN2A', 'APC', 'BRCA1', 'BRCA2', 'NF1',
    'VHL', 'WT1', 'STK11', 'SMAD4', 'ATM', 'CHEK2', 'MLH1',
    
    # -------------------------------------------------------------------------
    # DNA REPAIR
    # -------------------------------------------------------------------------
    'ERCC1', 'ERCC2', 'XRCC1', 'MGMT', 'MSH2', 'MSH6', 'PMS2',
    
    # -------------------------------------------------------------------------
    # APOPTOSIS
    # -------------------------------------------------------------------------
    'BCL2', 'BAX', 'CASP3', 'CASP8', 'CASP9', 'APAF1', 'XIAP',
    
    # -------------------------------------------------------------------------
    # CELL CYCLE
    # -------------------------------------------------------------------------
    'CCND1', 'CCNE1', 'CDK4', 'CDK6', 'CDKN1A', 'CDKN1B', 'CDC25A',
    
    # -------------------------------------------------------------------------
    # SIGNALING PATHWAYS
    # -------------------------------------------------------------------------
    'AKT1', 'AKT2', 'MTOR', 'MAPK1', 'MAPK3', 'JAK2', 'STAT3',
    'NOTCH1', 'WNT1', 'CTNNB1', 'SHH', 'SMO', 'GLI1',
    
    # -------------------------------------------------------------------------
    # DRUG METABOLISM & RESISTANCE
    # -------------------------------------------------------------------------
    'CYP1A2', 'CYP2D6', 'CYP3A4', 'ABCB1', 'ABCC1', 'ABCG2',
    
    # -------------------------------------------------------------------------
    # ANGIOGENESIS
    # -------------------------------------------------------------------------
    'VEGFA', 'KDR', 'HIF1A', 'ANGPT1', 'ANGPT2', 'TEK',
    
    # -------------------------------------------------------------------------
    # EPIGENETIC REGULATORS
    # -------------------------------------------------------------------------
    'DNMT1', 'DNMT3A', 'HDAC1', 'HDAC2', 'EZH2', 'KMT2A',
    
    # -------------------------------------------------------------------------
    # STEMNESS MARKERS
    # -------------------------------------------------------------------------
    'SOX2', 'NANOG', 'POU5F1', 'KLF4', 'ALDH1A1', 'CD44', 'PROM1',
    
    # -------------------------------------------------------------------------
    # METABOLISM
    # -------------------------------------------------------------------------
    'LDHA', 'PKM', 'HK2', 'G6PD', 'FASN', 'ACLY', 'SLC2A1',
    
    # -------------------------------------------------------------------------
    # EMT (Epithelial-Mesenchymal Transition)
    # -------------------------------------------------------------------------
    'CDH1', 'VIM', 'SNAI1', 'SNAI2', 'TWIST1', 'ZEB1', 'ZEB2',
    
    # -------------------------------------------------------------------------
    # HORMONE RECEPTORS
    # -------------------------------------------------------------------------
    'ESR1', 'AR', 'PGR', 'GATA3', 'FOXA1',
    
    # -------------------------------------------------------------------------
    # CELL PROLIFERATION
    # -------------------------------------------------------------------------
    'FOXM1', 'AURKA', 'AURKB', 'PLK1', 'BUB1', 'MAD2L1', 'MKI67',
    'TOP2A', 'TYMS', 'RRM1', 'RRM2', 'PCNA',

    # -------------------------------------------------------------------------
    # CANCER-ASSOCIATED FIBROBLASTS (CAFs)
    # -------------------------------------------------------------------------
    'FAP',      # Fibroblast Activation Protein - PRIMARY CAF MARKER
    'ACTA2',    # Alpha-smooth muscle actin (α-SMA)
    'PDGFRB',   # PDGF receptor beta
    'S100A4',   # FSP1 - fibroblast specific protein
    'PDPN',     # Podoplanin
    'COL1A1',   # Collagen type I
    'COL1A2',   # Collagen type I alpha 2
    'COL3A1',   # Collagen type III
    'FN1',      # Fibronectin
    'POSTN',    # Periostin - secreted by CAFs
    'THY1',     # CD90 - CAF subpopulation marker
    
    # -------------------------------------------------------------------------
    # T-CELL MARKERS
    # CD8A is critical for cytotoxic T-cell identification
    # -------------------------------------------------------------------------
    'CD8A',     # Cytotoxic T-cell marker - PRIMARY
    'CD8B',     # Cytotoxic T-cell marker
    'CD4',      # Helper T-cell marker
    'CD3D',     # T-cell receptor complex
    'CD3E',     # T-cell receptor complex
    'CD3G',     # T-cell receptor complex
    'FOXP3',    # Regulatory T-cell (Treg) marker
    'GZMB',     # Granzyme B - cytotoxic activity
    'PRF1',     # Perforin - cytotoxic activity
    'IFNG',     # Interferon gamma - T-cell activation
    'IL2',      # T-cell growth factor
    'CD28',     # T-cell co-stimulation
    'ICOS',     # Inducible T-cell co-stimulator
    
    # -------------------------------------------------------------------------
    # MACROPHAGE MARKERS (TAMs - Tumor Associated Macrophages)
    # -------------------------------------------------------------------------
    'CD68',     # Pan-macrophage marker
    'CD163',    # M2 macrophage marker
    'CSF1R',    # Colony stimulating factor 1 receptor
    'MARCO',    # M2 macrophage marker
    'MRC1',     # Mannose receptor (CD206)
    'CD14',     # Monocyte/macrophage marker
    'ITGAM',    # CD11b - myeloid marker
    'CD80',     # M1 macrophage marker
    'CD86',     # M1 macrophage marker
    'NOS2',     # iNOS - M1 macrophage marker
    'ARG1',     # Arginase 1 - M2 macrophage marker
    
    # -------------------------------------------------------------------------
    # IMMUNE CHECKPOINTS
    # Critical for immunotherapy response prediction
    # -------------------------------------------------------------------------
    'CD274',    # PD-L1
    'PDCD1',    # PD-1
    'PDCD1LG2', # PD-L2
    'CTLA4',    # CTLA-4
    'LAG3',     # LAG-3
    'HAVCR2',   # TIM-3
    'TIGIT',    # TIGIT
    'BTLA',     # BTLA
    'VSIR',     # VISTA
    'SIGLEC15', # Siglec-15
    'IDO1',     # Indoleamine 2,3-dioxygenase
    
    # -------------------------------------------------------------------------
    # IMMUNOSUPPRESSIVE TME SIGNALING
    # TGFB1 is a master regulator of immunosuppression
    # -------------------------------------------------------------------------
    'TGFB1',    # TGF-beta 1 - PRIMARY IMMUNOSUPPRESSIVE
    'TGFB2',    # TGF-beta 2
    'TGFB3',    # TGF-beta 3
    'IL10',     # Interleukin-10 - immunosuppressive
    'IL6',      # Interleukin-6 - pro-inflammatory/pro-tumor
    'CXCL12',   # SDF-1 - stromal derived factor
    'CXCR4',    # CXCL12 receptor
    'CCL2',     # MCP-1 - macrophage recruitment
    'CCL5',     # RANTES - immune cell recruitment
    'CXCL8',    # IL-8 - neutrophil recruitment
    'CXCL9',    # T-cell attractant
    'CXCL10',   # T-cell attractant
    'CXCL13',   # B-cell attractant
    
    # -------------------------------------------------------------------------
    # ANTIGEN PRESENTATION (MHC/HLA)
    # -------------------------------------------------------------------------
    'HLA-A',    # MHC class I
    'HLA-B',    # MHC class I
    'HLA-C',    # MHC class I
    'B2M',      # Beta-2-microglobulin
    'TAP1',     # Antigen peptide transporter
    'TAP2',     # Antigen peptide transporter
    'NLRC5',    # MHC class I transactivator
    'CIITA',    # MHC class II transactivator
    'HLA-DRA',  # MHC class II
    'HLA-DRB1', # MHC class II
    
    # -------------------------------------------------------------------------
    # NATURAL KILLER (NK) CELLS
    # -------------------------------------------------------------------------
    'NCR1',     # NKp46 - NK cell receptor
    'KLRK1',    # NKG2D
    'NCAM1',    # CD56 - NK cell marker
    'NKG7',     # NK cell granule protein
    'GNLY',     # Granulysin
    'KLRD1',    # CD94
    'KLRC1',    # NKG2A
    
    # -------------------------------------------------------------------------
    # DENDRITIC CELLS
    # -------------------------------------------------------------------------
    'CD1C',     # Conventional DC marker
    'CLEC9A',   # cDC1 marker
    'XCR1',     # cDC1 marker
    'BATF3',    # cDC1 transcription factor
    'IRF8',     # DC development
    'ITGAX',    # CD11c - DC marker
    'CD83',     # Mature DC marker
    'CCR7',     # DC migration
    
    # -------------------------------------------------------------------------
    # B-CELLS
    # -------------------------------------------------------------------------
    'CD19',     # B-cell marker
    'MS4A1',    # CD20
    'CD79A',    # B-cell receptor
    'PAX5',     # B-cell transcription factor
    
    # -------------------------------------------------------------------------
    # NEUTROPHILS
    # -------------------------------------------------------------------------
    'FCGR3B',   # CD16b
    'CEACAM8',  # CD66b
    'S100A8',   # Calprotectin subunit
    'S100A9',   # Calprotectin subunit
    
    # -------------------------------------------------------------------------
    # ENDOTHELIAL CELLS (Tumor Vasculature)
    # -------------------------------------------------------------------------
    'PECAM1',   # CD31
    'CDH5',     # VE-cadherin
    'VWF',      # Von Willebrand factor
    'ENG',      # Endoglin (CD105)
    'MCAM',     # CD146
]

# Real cancer cell line names from GDSC
CELL_LINES = [
    # Breast cancer
    'MCF7', 'MDA-MB-231', 'T47D', 'BT474', 'SKBR3', 'MDA-MB-468', 'BT549',
    'HCC1806', 'HCC1937', 'SUM149', 'SUM159', 'CAL51', 'HCC38', 'ZR751',
    'MDA-MB-453', 'MDA-MB-361', 'AU565', 'JIMT1', 'HCC1143', 'BT20',
    # Lung cancer
    'A549', 'H460', 'H1299', 'H522', 'H23', 'H358', 'HCC827', 'PC9',
    'H1975', 'H820', 'H1650', 'CALU1', 'CALU3', 'CALU6', 'H2228',
    # Colon cancer
    'HCT116', 'HT29', 'SW620', 'SW480', 'DLD1', 'LoVo', 'Caco2', 'RKO',
    'HCT15', 'COLO205', 'SW48', 'LS174T',
    # Ovarian cancer
    'SKOV3', 'OVCAR3', 'A2780', 'OVCAR8', 'ES2', 'IGROV1', 'OVCAR4',
    # Pancreatic cancer
    'PANC1', 'MIAPaCa2', 'BxPC3', 'AsPC1', 'Capan1', 'Capan2', 'CFPAC1',
    # Glioblastoma
    'U87MG', 'U251', 'T98G', 'LN229', 'A172', 'SF268', 'SF295',
    # Neuroblastoma
    'SKNSH', 'IMR32', 'KELLY', 'SHSY5Y', 'BE2C', 'SKNBE2', 'CHP212',
    # Leukemia
    'K562', 'HL60', 'U937', 'THP1', 'Jurkat', 'MOLT4', 'CCRF-CEM',
    # Melanoma
    'A375', 'SKMEL28', 'SKMEL5', 'MALME3M', 'LOXIMVI', 'UACC62', 'UACC257',
    # Renal cancer
    '786O', 'A498', 'Caki1', 'ACHN', 'RCC4', 'UMRC2',
    # Liver cancer
    'HepG2', 'Huh7', 'HCCLM3', 'PLC5', 'SNU449', 'SNU398', 'SNU387',
    # Prostate cancer
    'PC3', 'DU145', 'LNCaP', '22RV1', 'VCaP', 'MDAPCA2A',
]

# Common anticancer drugs from GDSC with TME-relevant mechanisms
DRUGS = {
    # DNA-damaging agents
    'Cisplatin': {
        'target': 'DNA crosslinking',
        'pathway': 'DNA damage',
        'biomarkers': ['ERCC1', 'BRCA1', 'BRCA2']
    },
    'Doxorubicin': {
        'target': 'TOP2A',
        'pathway': 'DNA damage',
        'biomarkers': ['TOP2A', 'ABCB1']
    },
    'Gemcitabine': {
        'target': 'RRM1/RRM2',
        'pathway': 'DNA synthesis',
        'biomarkers': ['RRM1', 'RRM2', 'DCK']
    },
    # Targeted therapies
    'Erlotinib': {
        'target': 'EGFR',
        'pathway': 'RTK signaling',
        'biomarkers': ['EGFR', 'KRAS', 'MET']
    },
    'Lapatinib': {
        'target': 'EGFR/ERBB2',
        'pathway': 'RTK signaling',
        'biomarkers': ['ERBB2', 'EGFR']
    },
    'Vemurafenib': {
        'target': 'BRAF V600E',
        'pathway': 'MAPK signaling',
        'biomarkers': ['BRAF', 'NRAS']
    },
    'Imatinib': {
        'target': 'BCR-ABL/KIT',
        'pathway': 'Kinase signaling',
        'biomarkers': ['ABL1', 'KIT', 'PDGFRA']
    },
    'Sorafenib': {
        'target': 'Multi-kinase (VEGFR, RAF)',
        'pathway': 'RTK/MAPK signaling',
        'biomarkers': ['VEGFA', 'KDR', 'BRAF']
    },
    # PARP inhibitors
    'Olaparib': {
        'target': 'PARP1/2',
        'pathway': 'DNA repair',
        'biomarkers': ['BRCA1', 'BRCA2', 'ATM']
    },
    # Microtubule agents
    'Paclitaxel': {
        'target': 'Microtubules',
        'pathway': 'Cell cycle',
        'biomarkers': ['TUBB3', 'BCL2', 'ABCB1']
    },
    'Docetaxel': {
        'target': 'Microtubules',
        'pathway': 'Cell cycle',
        'biomarkers': ['TUBB3', 'ABCB1']
    },
    # Immunotherapy (TME-relevant)
    'Pembrolizumab': {
        'target': 'PD-1',
        'pathway': 'Immune checkpoint',
        'biomarkers': ['CD274', 'PDCD1', 'CD8A', 'IFNG']
    },
    'Nivolumab': {
        'target': 'PD-1',
        'pathway': 'Immune checkpoint',
        'biomarkers': ['CD274', 'PDCD1', 'CD8A', 'TGFB1']
    },
    'Ipilimumab': {
        'target': 'CTLA-4',
        'pathway': 'Immune checkpoint',
        'biomarkers': ['CTLA4', 'CD8A', 'FOXP3']
    },
}


def load_sample_data(
    n_samples: int = 100,
    n_genes: int = 100,
    drug: str = 'Cisplatin',
    include_tme: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load sample cancer drug response data.
    
    Generates realistic gene expression and drug response data
    based on GDSC data characteristics, including TME markers.
    
    Parameters
    ----------
    n_samples : int, default=100
        Number of cell lines (samples)
    n_genes : int, default=100
        Number of genes to include
    drug : str, default='Cisplatin'
        Drug name for response prediction
    include_tme : bool, default=True
        Whether to prioritize TME genes in selection
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns
    -------
    X : np.ndarray of shape (n_samples, n_genes)
        Gene expression matrix (log2 normalized)
    y : np.ndarray of shape (n_samples,)
        Drug response labels (0=resistant, 1=sensitive)
    info : dict
        Metadata including gene names, cell lines, drug info
        
    Examples
    --------
    >>> from toporx.data import load_sample_data
    >>> X, y, info = load_sample_data(n_samples=100, n_genes=50)
    >>> print(f"Expression matrix: {X.shape}")
    >>> print(f"Genes: {info['gene_names'][:5]}")
    """
    np.random.seed(random_state)
    
    # Select genes - prioritize TME genes if requested
    gene_names = _select_genes(n_genes, include_tme)
    
    # Select cell lines
    n_samples = min(n_samples, len(CELL_LINES))
    cell_lines = CELL_LINES[:n_samples]
    
    # Generate realistic gene expression data
    X = _generate_expression_matrix(n_samples, len(gene_names), gene_names, random_state)
    
    # Generate drug response based on known biomarkers
    y = _generate_drug_response(X, gene_names, drug, random_state)
    
    # Compile metadata
    info = {
        'gene_names': gene_names,
        'cell_lines': cell_lines,
        'drug': drug,
        'drug_info': DRUGS.get(drug, {}),
        'n_sensitive': int(np.sum(y)),
        'n_resistant': int(np.sum(1 - y)),
        'n_tme_genes': _count_tme_genes(gene_names),
        'data_source': 'Simulated based on GDSC structure with TME markers',
        'expression_units': 'log2 normalized',
        'response_encoding': {0: 'Resistant', 1: 'Sensitive'}
    }
    
    return X, y, info


def _select_genes(n_genes: int, include_tme: bool) -> list:
    """
    Select genes, prioritizing TME markers if requested.
    """
    # Key TME genes to always include
    tme_priority = [
        'FAP', 'CD8A', 'TGFB1',  # Primary TME markers
        'CD274', 'PDCD1', 'CTLA4',  # Immune checkpoints
        'CD68', 'CD163',  # Macrophages
        'FOXP3', 'GZMB',  # T-cell subsets
        'VEGFA', 'HIF1A',  # Angiogenesis
        'COL1A1', 'ACTA2',  # CAFs
        'IFNG', 'IL10', 'IL6',  # Cytokines
    ]
    
    # Key cancer genes to always include
    cancer_priority = [
        'TP53', 'EGFR', 'KRAS', 'BRAF', 'PIK3CA',
        'BRCA1', 'BRCA2', 'PTEN', 'MYC', 'ERBB2',
        'BCL2', 'ABCB1', 'TOP2A', 'RRM1', 'ERCC1'
    ]
    
    if include_tme:
        # Start with TME genes, then cancer genes, then fill rest
        selected = []
        for gene in tme_priority:
            if gene in CANCER_GENES and gene not in selected:
                selected.append(gene)
                if len(selected) >= n_genes:
                    return selected
        
        for gene in cancer_priority:
            if gene in CANCER_GENES and gene not in selected:
                selected.append(gene)
                if len(selected) >= n_genes:
                    return selected
        
        # Fill remaining from full list
        for gene in CANCER_GENES:
            if gene not in selected:
                selected.append(gene)
                if len(selected) >= n_genes:
                    return selected
        
        return selected
    else:
        # Just take first n genes
        return CANCER_GENES[:n_genes]


def _count_tme_genes(gene_names: list) -> int:
    """Count TME-related genes in the selection."""
    tme_genes = {
        'FAP', 'ACTA2', 'PDGFRB', 'S100A4', 'PDPN', 'COL1A1', 'COL3A1',
        'CD8A', 'CD8B', 'CD4', 'CD3D', 'CD3E', 'FOXP3', 'GZMB', 'PRF1',
        'CD68', 'CD163', 'CSF1R', 'MARCO', 'MRC1', 'CD14',
        'CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'IDO1',
        'TGFB1', 'TGFB2', 'IL10', 'IL6', 'CXCL12', 'CCL2',
        'HLA-A', 'HLA-B', 'HLA-C', 'B2M', 'TAP1', 'TAP2'
    }
    return sum(1 for g in gene_names if g in tme_genes)


def _generate_expression_matrix(
    n_samples: int,
    n_genes: int,
    gene_names: list,
    random_state: int
) -> np.ndarray:
    """
    Generate realistic gene expression matrix.
    
    Creates log2 normalized expression values with:
    - Gene-specific baseline expression
    - Sample-specific effects (tumor heterogeneity)
    - Realistic correlation structure (co-expression modules)
    - TME-specific patterns
    """
    np.random.seed(random_state)
    
    # Gene-specific baseline (some genes highly expressed, others low)
    gene_baselines = {}
    for gene in gene_names:
        # TME genes often have specific expression patterns
        if gene in ['CD8A', 'GZMB', 'PRF1', 'IFNG']:
            # T-cell genes: bimodal (high in immune-hot tumors)
            gene_baselines[gene] = np.random.choice([4.0, 9.0])
        elif gene in ['FAP', 'ACTA2', 'COL1A1']:
            # CAF genes: moderate to high
            gene_baselines[gene] = np.random.uniform(6, 10)
        elif gene in ['TGFB1', 'IL6', 'IL10']:
            # Cytokines: moderate expression
            gene_baselines[gene] = np.random.uniform(5, 8)
        elif gene in ['CD274', 'PDCD1', 'CTLA4']:
            # Checkpoints: typically lower
            gene_baselines[gene] = np.random.uniform(3, 6)
        elif gene in ['MYC', 'CCND1', 'EGFR']:
            # Oncogenes: often high in cancer
            gene_baselines[gene] = np.random.uniform(7, 11)
        elif gene in ['TP53', 'BRCA1', 'BRCA2']:
            # Tumor suppressors: variable
            gene_baselines[gene] = np.random.uniform(5, 9)
        else:
            # Other genes: random baseline
            gene_baselines[gene] = np.random.uniform(4, 10)
    
    # Gene-specific variance
    gene_variance = {gene: np.random.uniform(0.5, 1.5) for gene in gene_names}
    
    X = np.zeros((n_samples, n_genes))
    
    # Create sample subtypes (immune hot vs cold, high vs low stroma)
    immune_hot = np.random.random(n_samples) > 0.5
    high_stroma = np.random.random(n_samples) > 0.6
    
    for i in range(n_samples):
        # Sample-specific global effect
        sample_effect = np.random.normal(0, 0.3)
        
        for j, gene in enumerate(gene_names):
            base = gene_baselines[gene]
            var = gene_variance[gene]
            
            # Adjust based on sample phenotype
            if immune_hot[i] and gene in ['CD8A', 'GZMB', 'PRF1', 'IFNG', 'CXCL9', 'CXCL10']:
                base += 2.5  # Higher in immune-hot tumors
            elif not immune_hot[i] and gene in ['TGFB1', 'IL10']:
                base += 1.5  # Higher immunosuppression in cold tumors
            
            if high_stroma[i] and gene in ['FAP', 'ACTA2', 'COL1A1', 'COL3A1', 'FN1']:
                base += 2.0  # Higher in stroma-rich tumors
            
            X[i, j] = base + sample_effect + np.random.normal(0, var)
    
    # Clip to realistic range (2-15 for log2 expression)
    X = np.clip(X, 2, 15)
    
    return X


def _generate_drug_response(
    X: np.ndarray,
    gene_names: list,
    drug: str,
    random_state: int
) -> np.ndarray:
    """
    Generate drug response labels based on expression biomarkers.
    
    Uses known drug-gene relationships including TME factors
    for immunotherapy response.
    """
    np.random.seed(random_state)
    
    n_samples = X.shape[0]
    
    # Drug-specific biomarkers including TME genes for immunotherapy
    drug_biomarkers = {
        'Cisplatin': {
            'sensitive_low': ['ERCC1', 'ERCC2'],  # Low = sensitive
            'sensitive_high': [],
            'resistant_high': ['ABCC1', 'ABCG2']  # High = resistant
        },
        'Paclitaxel': {
            'sensitive_low': ['BCL2', 'TUBB3'],
            'sensitive_high': [],
            'resistant_high': ['ABCB1']
        },
        'Erlotinib': {
            'sensitive_low': [],
            'sensitive_high': ['EGFR'],
            'resistant_high': ['MET', 'KRAS']
        },
        'Vemurafenib': {
            'sensitive_low': [],
            'sensitive_high': ['BRAF'],
            'resistant_high': ['NRAS', 'MAP2K1']
        },
        'Olaparib': {
            'sensitive_low': ['BRCA1', 'BRCA2'],  # BRCA deficient = sensitive
            'sensitive_high': [],
            'resistant_high': ['ABCB1']
        },
        'Doxorubicin': {
            'sensitive_low': [],
            'sensitive_high': ['TOP2A'],
            'resistant_high': ['ABCB1', 'ABCC1']
        },
        # Immunotherapy - TME genes are critical!
        'Pembrolizumab': {
            'sensitive_low': ['TGFB1'],  # Low TGFB1 = better response
            'sensitive_high': ['CD8A', 'GZMB', 'IFNG', 'CD274'],  # Immune hot = sensitive
            'resistant_high': ['FOXP3', 'IL10']  # Immunosuppressive = resistant
        },
        'Nivolumab': {
            'sensitive_low': ['TGFB1'],
            'sensitive_high': ['CD8A', 'GZMB', 'PRF1', 'CD274'],
            'resistant_high': ['FOXP3', 'IL10', 'LAG3']
        },
        'Ipilimumab': {
            'sensitive_low': [],
            'sensitive_high': ['CD8A', 'CTLA4', 'IFNG'],
            'resistant_high': ['TGFB1', 'IL10', 'FOXP3']
        },
    }
    
    # Get biomarkers for selected drug
    biomarkers = drug_biomarkers.get(drug, {
        'sensitive_low': ['ERCC1'],
        'sensitive_high': [],
        'resistant_high': ['ABCB1']
    })
    
    # Calculate sensitivity score
    sensitivity_score = np.zeros(n_samples)
    
    # Low expression → sensitive
    for gene in biomarkers.get('sensitive_low', []):
        if gene in gene_names:
            idx = gene_names.index(gene)
            sensitivity_score -= X[:, idx] * 0.5  # Lower = more sensitive
    
    # High expression → sensitive
    for gene in biomarkers.get('sensitive_high', []):
        if gene in gene_names:
            idx = gene_names.index(gene)
            sensitivity_score += X[:, idx] * 0.5  # Higher = more sensitive
    
    # High expression → resistant
    for gene in biomarkers.get('resistant_high', []):
        if gene in gene_names:
            idx = gene_names.index(gene)
            sensitivity_score -= X[:, idx] * 0.3  # Higher = more resistant
    
    # Normalize and add noise
    sensitivity_score = (sensitivity_score - sensitivity_score.mean()) / (sensitivity_score.std() + 1e-6)
    sensitivity_score += np.random.normal(0, 0.5, n_samples)
    
    # Convert to binary (aim for ~40-60% sensitive)
    threshold = np.percentile(sensitivity_score, 50)
    y = (sensitivity_score > threshold).astype(int)
    
    return y


def load_gdsc_subset(
    cancer_type: str = 'breast',
    drug: str = 'Cisplatin',
    include_tme: bool = True,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load subset mimicking specific GDSC cancer type.
    
    Parameters
    ----------
    cancer_type : str
        Cancer type: 'breast', 'lung', 'colon', 'pancreas', 'melanoma',
        'glioblastoma', 'ovarian'
    drug : str
        Drug name
    include_tme : bool
        Include TME genes
    random_state : int
        Random seed
        
    Returns
    -------
    X, y, info : Same as load_sample_data
    """
    # Cancer-specific configurations
    cancer_configs = {
        'breast': {'n_samples': 80, 'n_genes': 100, 'drugs': ['Paclitaxel', 'Doxorubicin', 'Lapatinib']},
        'lung': {'n_samples': 120, 'n_genes': 100, 'drugs': ['Cisplatin', 'Erlotinib', 'Pembrolizumab']},
        'colon': {'n_samples': 60, 'n_genes': 100, 'drugs': ['Cisplatin', 'Pembrolizumab']},
        'pancreas': {'n_samples': 40, 'n_genes': 100, 'drugs': ['Gemcitabine', 'Olaparib']},
        'melanoma': {'n_samples': 50, 'n_genes': 100, 'drugs': ['Vemurafenib', 'Pembrolizumab', 'Ipilimumab']},
        'glioblastoma': {'n_samples': 45, 'n_genes': 100, 'drugs': ['Cisplatin', 'Sorafenib']},
        'ovarian': {'n_samples': 55, 'n_genes': 100, 'drugs': ['Cisplatin', 'Olaparib', 'Paclitaxel']}
    }
    
    config = cancer_configs.get(cancer_type, {'n_samples': 100, 'n_genes': 100, 'drugs': ['Cisplatin']})
    
    X, y, info = load_sample_data(
        n_samples=config['n_samples'],
        n_genes=config['n_genes'],
        drug=drug,
        include_tme=include_tme,
        random_state=random_state
    )
    
    info['cancer_type'] = cancer_type
    info['recommended_drugs'] = config['drugs']
    
    return X, y, info


def get_data_info() -> Dict:
    """
    Get information about available data.
    
    Returns
    -------
    dict
        Available genes, cell lines, drugs, and TME markers
    """
    tme_genes = [
        'FAP', 'CD8A', 'TGFB1', 'CD274', 'PDCD1', 'CTLA4',
        'CD68', 'CD163', 'FOXP3', 'GZMB', 'ACTA2', 'COL1A1'
    ]
    
    return {
        'n_available_genes': len(CANCER_GENES),
        'n_available_cell_lines': len(CELL_LINES),
        'available_drugs': list(DRUGS.keys()),
        'tme_genes': tme_genes,
        'immunotherapy_drugs': ['Pembrolizumab', 'Nivolumab', 'Ipilimumab'],
        'gene_list': CANCER_GENES,
        'cell_line_list': CELL_LINES,
        'drug_info': DRUGS
    }


def get_gene_info(gene: str) -> Dict:
    """
    Get information about a specific gene.
    
    Parameters
    ----------
    gene : str
        Gene symbol (e.g., 'FAP', 'CD8A', 'TGFB1')
        
    Returns
    -------
    dict
        Gene information including category and TME relevance
    """
    gene_categories = {
        'oncogenes': ['EGFR', 'KRAS', 'BRAF', 'PIK3CA', 'MYC', 'ERBB2', 'MET', 'ALK'],
        'tumor_suppressors': ['TP53', 'RB1', 'PTEN', 'CDKN2A', 'APC', 'BRCA1', 'BRCA2'],
        'dna_repair': ['ERCC1', 'ERCC2', 'XRCC1', 'MGMT', 'MSH2', 'MSH6', 'BRCA1', 'BRCA2'],
        'drug_resistance': ['ABCB1', 'ABCC1', 'ABCG2'],
        'apoptosis': ['BCL2', 'BAX', 'CASP3', 'CASP8', 'CASP9'],
        'tme_cafs': ['FAP', 'ACTA2', 'PDGFRB', 'S100A4', 'COL1A1', 'COL3A1'],
        'tme_tcells': ['CD8A', 'CD8B', 'CD4', 'CD3D', 'FOXP3', 'GZMB', 'PRF1'],
        'tme_macrophages': ['CD68', 'CD163', 'CSF1R', 'MARCO', 'MRC1'],
        'tme_checkpoints': ['CD274', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT'],
        'tme_immunosuppressive': ['TGFB1', 'TGFB2', 'IL10', 'IL6', 'IDO1']
    }
    
    categories = []
    is_tme = False
    
    for category, genes in gene_categories.items():
        if gene in genes:
            categories.append(category)
            if category.startswith('tme_'):
                is_tme = True
    
    return {
        'gene': gene,
        'in_dataset': gene in CANCER_GENES,
        'categories': categories if categories else ['other'],
        'is_tme_gene': is_tme,
        'description': _get_gene_description(gene)
    }


def _get_gene_description(gene: str) -> str:
    """Get brief description of gene function."""
    descriptions = {
        'FAP': 'Fibroblast Activation Protein - CAF marker, TME remodeling',
        'CD8A': 'Cytotoxic T-cell marker - immune infiltration',
        'TGFB1': 'TGF-beta 1 - immunosuppressive signaling',
        'CD274': 'PD-L1 - immune checkpoint ligand',
        'PDCD1': 'PD-1 - immune checkpoint receptor',
        'CTLA4': 'CTLA-4 - immune checkpoint receptor',
        'CD68': 'Pan-macrophage marker',
        'CD163': 'M2 macrophage marker - immunosuppressive TAMs',
        'FOXP3': 'Regulatory T-cell (Treg) marker',
        'GZMB': 'Granzyme B - cytotoxic activity marker',
        'EGFR': 'Epidermal growth factor receptor - oncogene',
        'TP53': 'Tumor protein p53 - tumor suppressor',
        'BRCA1': 'BRCA1 DNA repair - tumor suppressor',
        'ABCB1': 'P-glycoprotein - drug efflux pump',
    }
    return descriptions.get(gene, 'Cancer-related gene')
