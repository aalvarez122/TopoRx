<h1 align="center">TopoRx</h1>

<p align="center">
  <strong>Predicting Cancer Drug Response Using Topological Data Analysis</strong>
</p>

<p align="center">
  <em>When the shape of your data predicts survival</em>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/TDA-Persistent_Homology-purple.svg" alt="TDA"></a>
  <a href="#"><img src="https://img.shields.io/badge/TME-Tumor_Microenvironment-orange.svg" alt="TME"></a>
</p>

---

##  What is TopoRx?

**TopoRx** applies **Topological Data Analysis (TDA)** to predict how cancer patients respond to drugs. Instead of looking at individual genes, we analyze the *shape* of gene expression data to discover biomarkers invisible to traditional methods.

### The Problem

Traditional biomarker discovery:
- Focuses on single genes → misses complex interactions
- Sensitive to noise → unreliable predictions
- Linear methods → can't capture non-linear biology

### Our Solution
```
Gene Expression → Topological Features → Drug Response Prediction
                        ↓
            • Persistent Homology (Ripser)
            • Persistence Landscapes  
            • Betti Curves
```

**Result**: Topological biomarkers that capture the *shape* of disease, with potential to improve drug response prediction over traditional gene-only methods.

---

##  The Science

### Why Topology?

Cancer is complex. The relationship between genes, proteins, and drug response isn't linear—it's a **high-dimensional shape**. TDA captures this shape through:

| Concept | What it captures | Biological meaning |
|---------|------------------|-------------------|
| **H0 (Connected Components)** | Clusters in data | Distinct patient subgroups |
| **H1 (Loops)** | Circular patterns | Feedback loops, gene cycles |
| **H2 (Voids)** | Cavities in data | Missing interactions |

### Persistence = Importance

Features that **persist** across multiple scales are real signals, not noise. This makes TDA inherently robust to biological variability.

### Tumor Microenvironment (TME) Integration

TopoRx includes key TME genes relevant for immunotherapy response prediction:

| Gene | Role | Relevance |
|------|------|-----------|
| **FAP** | Cancer-associated fibroblast marker | TME remodeling |
| **CD8A** | Cytotoxic T-cell marker | Immune infiltration |
| **TGFB1** | Immunosuppressive signaling | Treatment resistance |
| **CD274** | PD-L1 | Checkpoint inhibitor target |

---

##  Quick Start

### Installation
```bash
git clone https://github.com/aalvarez122/TopoRx.git
cd TopoRx
pip install -r requirements.txt
```

### Basic Usage
```python
from toporx.data import load_sample_data
from toporx.tda import PersistentHomologyComputer, TopologicalFeatureExtractor
from toporx.prediction import DrugResponseClassifier
import numpy as np

# Load sample cancer data (includes TME genes)
X, y, info = load_sample_data(
    n_samples=100,
    n_genes=80,
    drug='Pembrolizumab',  # PD-1 inhibitor
    include_tme=True
)

print(f"Loaded: {X.shape[0]} samples, {X.shape[1]} genes")
print(f"TME genes included: {info['n_tme_genes']}")

# Compute persistent homology with Ripser
ph = PersistentHomologyComputer(max_dimension=2, metric='correlation')
diagrams = ph.fit_transform(X)

# View topological summary
print(ph.summary())

# Extract ML-ready features
extractor = TopologicalFeatureExtractor(
    feature_types=["statistics", "entropy", "betti"]
)
topo_features = extractor.fit_transform(diagrams)

print(f"Extracted {len(topo_features)} topological features")
```

### Run Demo
```bash
python examples/quick_start.py
```

**Expected output:**
```
 TopoRx: Topological Biomarker Discovery
   Predicting Cancer Drug Response Using TDA
===========================================================

 STEP 1: Loading Cancer Drug Response Data
   ✓ Loaded 100 samples × 80 genes
   ✓ Drug: Pembrolizumab
   ✓ TME genes included: 12

 STEP 2: Computing Persistent Homology
   → H0: 45 features, max persistence = 0.8234
   → H1: 23 features, max persistence = 0.4521
   → H2: 8 features, max persistence = 0.2103

 RESULTS
   Gene Expression Only:    ROC-AUC = 0.XXX
   Topological Features:    ROC-AUC = 0.XXX
   Combined (TDA + Genes):  ROC-AUC = 0.XXX
```

---

##  Features

### Topological Feature Extraction
- **Persistent Homology** — Compute H0, H1, H2 features using Ripser
- **Persistence Landscapes** — Functional summaries for statistical analysis
- **Betti Curves** — Track topological features across filtration
- **Persistence Entropy** — Information-theoretic topological summary

### Machine Learning Integration
- Seamless scikit-learn integration
- Built-in cross-validation
- Comparison with traditional gene-based methods
- Feature importance analysis

### Interactive Visualizations
- Persistence diagrams (Plotly)
- Betti curve plots
- Feature importance charts
- Model comparison visualizations

### TME-Aware Data
- 200+ cancer-related genes
- Tumor microenvironment markers (FAP, CD8A, TGFB1)
- Immunotherapy biomarkers
- Multiple drug response datasets

---

##  Project Structure
```
TopoRx/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── toporx/                     # Main package
│   ├── __init__.py
│   ├── pipeline.py             # Main pipeline class
│   │
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   └── loader.py           # Sample data with TME genes
│   │
│   ├── tda/                    # Topological Data Analysis
│   │   ├── __init__.py
│   │   ├── persistence.py      # Ripser-based persistent homology
│   │   ├── features.py         # Topological feature extraction
│   │   └── landscapes.py       # Persistence landscapes
│   │
│   ├── prediction/             # ML models
│   │   ├── __init__.py
│   │   ├── classifier.py       # Drug response classifier
│   │   └── evaluation.py       # Model evaluation metrics
│   │
│   └── visualization/          # Plotting
│       ├── __init__.py
│       └── plots.py            # Plotly visualizations
│
└── examples/
    └── quick_start.py          # Demo script
```

---

##  Biological Context

### Why This Matters for Cancer Treatment

1. **Precision Medicine**: Match patients to drugs based on topological signatures
2. **Biomarker Discovery**: Find robust markers that survive noise
3. **Immunotherapy Prediction**: TME genes influence checkpoint inhibitor response
4. **Drug Resistance**: Understand treatment failure mechanisms

### Supported Drugs

| Drug | Target | Type |
|------|--------|------|
| Pembrolizumab | PD-1 | Immunotherapy |
| Nivolumab | PD-1 | Immunotherapy |
| Cisplatin | DNA | Chemotherapy |
| Paclitaxel | Microtubules | Chemotherapy |
| Erlotinib | EGFR | Targeted |
| Vemurafenib | BRAF | Targeted |
| Olaparib | PARP | Targeted |

---

##  Limitations

This is a **demonstration project** for educational and portfolio purposes:

- Uses simulated data based on GDSC structure (not actual patient data)
- Performance metrics will vary with random seeds
- Not validated for clinical use
- Simplified per-sample TDA computation

For production use, consider:
- Real clinical datasets (GDSC, CCLE, TCGA)
- Per-patient persistent homology computation
- Rigorous cross-study validation

---

##  Contributing

Contributions welcome! Areas for improvement:

- [ ] Add real GDSC data integration
- [ ] Per-sample persistence computation
- [ ] Additional TDA features (persistence images)
- [ ] Jupyter notebook tutorials
- [ ] Unit tests
```bash
# Development setup
git clone https://github.com/aalvarez122/TopoRx.git
cd TopoRx
pip install -r requirements.txt
python examples/quick_start.py  # Verify it works
```

---

##  License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Author

**Angelica Alvarez**
-  B.S. Neuroscience, University of Arizona

[![GitHub](https://img.shields.io/badge/GitHub-aalvarez122-black)](https://github.com/aalvarez122)

---

##  Acknowledgments

- [Ripser](https://ripser.scikit-tda.org/) — Fast persistent homology computation
- [scikit-learn](https://scikit-learn.org/) — Machine learning framework
- [Plotly](https://plotly.com/) — Interactive visualizations
- [GDSC](https://www.cancerrxgene.org/) — Data structure inspiration

---

<p align="center">
  <strong>If topology can describe the shape of the universe, it can describe the shape of disease.</strong>
</p>
