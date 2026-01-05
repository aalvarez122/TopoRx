<p align="center">
  <img src="assets/logo.png" alt="TopoRx Logo" width="200">
</p>

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
  <a href="#"><img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status"></a>
</p>

---

##  What is TopoRx?

**TopoRx** applies cutting-edge **Topological Data Analysis (TDA)** to predict how cancer patients respond to drugs. Instead of looking at individual genes, we analyze the *shape* of gene expression data to discover biomarkers invisible to traditional methods.

### The Problem

Traditional biomarker discovery:
- Focuses on single genes → misses complex interactions
- Sensitive to noise → unreliable predictions
- Linear methods → can't capture non-linear biology

### Our Solution

```
Gene Expression → Topological Features → Drug Response Prediction
                        ↓
            • Persistent Homology
            • Persistence Landscapes  
            • Betti Curves
---

##  The Science

### Why Topology?

Cancer is complex. The relationship between genes, proteins, and drug response isn't linear—it's a high-dimensional shape. TDA captures this shape through:

| Concept | What it captures | Biological meaning |
|---------|------------------|-------------------|
| **H0 (Connected Components)** | Clusters in data | Distinct patient subgroups |
| **H1 (Loops)** | Circular patterns | Feedback loops, cycles |
| **H2 (Voids)** | Cavities in data | Missing interactions |

### Persistence = Importance

Features that **persist** across multiple scales are real signals, not noise. This makes TDA inherently robust.

<p align="center">
  <img src="assets/persistence_diagram.png" alt="Persistence Diagram" width="600">
</p>

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
from toporx import TopoRxPipeline

# Initialize pipeline
pipeline = TopoRxPipeline()

# Load your gene expression data
pipeline.load_data(
    expression_matrix="data/expression.csv",
    drug_response="data/response.csv"
)

# Extract topological features
topo_features = pipeline.extract_topological_features()

# Train drug response predictor
results = pipeline.train_predictor()

# Visualize results
pipeline.plot_persistence_diagram()
pipeline.plot_feature_importance()
```

### One-Line Demo

```python
from toporx import demo
demo.run()  # Runs full analysis on sample data
```

---

##  Features

### Topological Feature Extraction
- **Persistent Homology** — Compute H0, H1, H2 features
- **Persistence Landscapes** — Functional summaries of persistence
- **Betti Curves** — Track topological features across filtration
- **Persistence Entropy** — Information-theoretic summary
- **Persistence Images** — CNN-ready representations

### Machine Learning Integration
- Seamless sklearn integration
- Built-in cross-validation
- Comparison with traditional gene-based methods
- Feature importance analysis

### Interactive Visualizations (Plotly)
- 3D persistence diagrams
- Interactive Betti curves
- Drug response prediction dashboard
- Feature importance plots

##  Project Structure

```
TopoRx/
├── README.md
├── requirements.txt
├── LICENSE
├── setup.py
│
├── toporx/                     # Main package
│   ├── __init__.py
│   ├── pipeline.py             # Main TopoRxPipeline class
│   │
│   ├── data/                   # Data handling
│   │   ├── __init__.py
│   │   ├── loader.py           # Data loading utilities
│   │   └── sample_data/        # Sample datasets
│   │
│   ├── tda/                    # Topological Data Analysis
│   │   ├── __init__.py
│   │   ├── persistence.py      # Persistent homology
│   │   ├── landscapes.py       # Persistence landscapes
│   │   ├── features.py         # Feature extraction
│   │   └── filtrations.py      # Filtration methods
│   │
│   ├── prediction/             # ML models
│   │   ├── __init__.py
│   │   ├── classifier.py       # Drug response classifier
│   │   └── evaluation.py       # Model evaluation
│   │
│   └── visualization/          # Plotting
│       ├── __init__.py
│       ├── persistence_plots.py
│       ├── landscapes_plot.py
│       └── dashboard.py
│
├── notebooks/
│   └── demo.ipynb              # Interactive tutorial
│
├── examples/
│   ├── quick_start.py
│   └── full_analysis.py
│
└── tests/
    └── test_pipeline.py
```

---

### Why This Matters for Cancer Treatment

1. **Precision Medicine**: Match patients to drugs based on their topological signature
2. **Biomarker Discovery**: Find robust markers that survive noise
3. **Drug Resistance**: Understand why some patients don't respond
4. **Clinical Trials**: Better patient stratification

### Potential Applications

- Pre-clinical drug screening
-  Clinical trial patient selection
-  Drug repurposing
-  Resistance mechanism discovery


##  Contributing

Contributions welcome! Please read our contributing guidelines first.

```bash
# Development setup
git clone https://github.com/aalvarez122/TopoRx.git
cd TopoRx
pip install -e ".[dev]"
pytest tests/
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

##  Author

**Angelica Alvarez**

 B.S. Neuroscience & Cognitive Science, University of Arizona

[![GitHub](https://img.shields.io/badge/GitHub-aalvarez122-black)](https://github.com/aalvarez122)

---

##  Acknowledgments

- Built with [GUDHI](https://gudhi.inria.fr/), [scikit-learn](https://scikit-learn.org/), and [Plotly](https://plotly.com/)

---

<p align="center">
  <strong>If topology can describe the shape of the universe, it can describe the shape of disease.</strong>
</p>
