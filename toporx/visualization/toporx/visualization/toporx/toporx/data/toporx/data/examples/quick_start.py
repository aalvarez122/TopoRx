#!/usr/bin/env python
"""
TopoRx Quick Start Example
==========================

Demonstrates the TopoRx pipeline for predicting cancer drug
response using Topological Data Analysis (TDA).

The purpose is to show the pipeline workflow, not to
make claims about TDA vs gene-based prediction performance.

For real research, use actual GDSC/CCLE data and proper
per-sample topological analysis.

Author: Angelica Alvarez
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    """Run quick start demo."""
    
    print()
    print("=" * 65)
    print(" TopoRx: Topological Biomarker Discovery")
    print("   Predicting Cancer Drug Response Using TDA")
    print("=" * 65)
    print()
    print("  NOTE: This demo uses SYNTHETIC data for demonstration.")
    print("    Results shown are for pipeline demonstration only.")
    print()
    
    # -----------------------------------------------------------------
    # Step 1: Load synthetic cancer data
    # -----------------------------------------------------------------
    print(" STEP 1: Loading Synthetic Cancer Drug Response Data")
    print("-" * 55)
    
    from toporx.data import load_sample_data, get_gene_info
    
    # Load synthetic data - includes TME genes
    X, y, info = load_sample_data(
        n_samples=100,
        n_genes=80,
        drug='Pembrolizumab',  # PD-1 inhibitor (immunotherapy)
        include_tme=True,
        random_state=42
    )
    
    print(f"   ✓ Loaded {X.shape[0]} samples × {X.shape[1]} genes")
    print(f"   ✓ Drug: {info['drug']}")
    print(f"   ✓ Sensitive: {info['n_sensitive']} | Resistant: {info['n_resistant']}")
    print(f"   ✓ TME genes included: {info['n_tme_genes']}")
    print(f"     Data source: {info['data_source']}")
    
    # Show key TME genes
    print(f"\n   Key TME genes in dataset:")
    for gene in ['FAP', 'CD8A', 'TGFB1']:
        if gene in info['gene_names']:
            gene_info = get_gene_info(gene)
            print(f"   • {gene}: {gene_info['description']}")
    
    # -----------------------------------------------------------------
    # Step 2: Compute Persistent Homology
    # -----------------------------------------------------------------
    print("\n STEP 2: Computing Persistent Homology (Ripser)")
    print("-" * 55)
    
    from toporx.tda import PersistentHomologyComputer
    
    ph = PersistentHomologyComputer(
        max_dimension=2,
        metric='correlation'
    )
    diagrams = ph.fit_transform(X)
    
    print("\n   Topological Summary (Patient-Space TDA):")
    summary = ph.summary()
    for dim, stats in summary.items():
        print(f"   • {dim}: {stats['n_features']} features, "
              f"max persistence = {stats['max_persistence']:.4f}")
    
    # -----------------------------------------------------------------
    # Step 3: Extract Topological Features
    # -----------------------------------------------------------------
    print("\n STEP 3: Extracting Topological Features")
    print("-" * 55)
    
    from toporx.tda import TopologicalFeatureExtractor
    
    extractor = TopologicalFeatureExtractor(
        feature_types=["statistics", "entropy", "betti"]
    )
    topo_features = extractor.fit_transform(diagrams)
    feature_names = extractor.get_feature_names()
    
    print(f"   ✓ Extracted {len(topo_features)} topological features")
    print(f"   ✓ Feature types: statistics, entropy, betti curves")
    
    # -----------------------------------------------------------------
    # Step 4: Build Feature Matrix
    # -----------------------------------------------------------------
    print("\n STEP 4: Building Feature Matrix")
    print("-" * 55)
    
    # IMPORTANT: This is a SIMPLIFIED approach for demonstration
    # 
    # Current approach:
    #   - Compute ONE persistence diagram from all samples
    #   - Create per-sample features by adding noise
    #   - This is NOT proper per-sample TDA
    #
    # Proper approach (for real research):
    #   - Compute persistence for each sample individually
    #   - OR use patient-neighborhood based TDA
    #   - OR use time-series TDA approaches
    #
    # We use the simplified approach here to demonstrate
    # the pipeline without excessive computation time.
    
    np.random.seed(42)
    n_samples = X.shape[0]
    
    # Create sample-specific features (SIMPLIFIED - adds noise variation)
    # In production, compute proper per-sample TDA
    X_topo = np.array([
        topo_features + np.random.randn(len(topo_features)) * 0.1
        for _ in range(n_samples)
    ])
    
    print(f"   ✓ Topological features: {X_topo.shape}")
    print(f"   ✓ Gene expression: {X.shape}")
    print(f"     Note: Using simplified per-sample features for demo")
    
    # -----------------------------------------------------------------
    # Step 5: Train and Evaluate Classifier
    # -----------------------------------------------------------------
    print("\n STEP 5: Training Drug Response Classifier")
    print("-" * 55)
    
    from toporx.prediction import DrugResponseClassifier
    
    # Evaluate TDA features
    clf_topo = DrugResponseClassifier(model_type="random_forest", random_state=42)
    cv_topo = clf_topo.cross_validate(X_topo, y, cv=5, scoring='roc_auc')
    
    # Evaluate gene features (traditional approach)
    clf_genes = DrugResponseClassifier(model_type="random_forest", random_state=42)
    cv_genes = clf_genes.cross_validate(X, y, cv=5, scoring='roc_auc')
    
    # Evaluate combined features
    X_combined = np.hstack([X_topo, X])
    clf_combined = DrugResponseClassifier(model_type="random_forest", random_state=42)
    cv_combined = clf_combined.cross_validate(X_combined, y, cv=5, scoring='roc_auc')
    
    print(f"   • TDA Features:  ROC-AUC = {cv_topo['mean_score']:.3f} ± {cv_topo['std_score']:.3f}")
    print(f"   • Gene Features: ROC-AUC = {cv_genes['mean_score']:.3f} ± {cv_genes['std_score']:.3f}")
    print(f"   • Combined:      ROC-AUC = {cv_combined['mean_score']:.3f} ± {cv_combined['std_score']:.3f}")
    
    # -----------------------------------------------------------------
    # Step 6: Results Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 65)
    print(" RESULTS (Demonstration Only)")
    print("=" * 65)
    print()
    print("   ┌───────────────────────────────────────────────────────┐")
    print("   │      DRUG RESPONSE PREDICTION - PIPELINE DEMO        │")
    print("   ├───────────────────────────────────────────────────────┤")
    print(f"   │   Gene Expression:    ROC-AUC = {cv_genes['mean_score']:.3f} ± {cv_genes['std_score']:.3f}       │")
    print(f"   │   TDA Features:       ROC-AUC = {cv_topo['mean_score']:.3f} ± {cv_topo['std_score']:.3f}       │")
    print(f"   │   Combined:           ROC-AUC = {cv_combined['mean_score']:.3f} ± {cv_combined['std_score']:.3f}       │")
    print("   ├───────────────────────────────────────────────────────┤")
    print("   │     These results are for DEMONSTRATION only.      │")
    print("   │   Real performance requires actual clinical data.    │")
    print("   └───────────────────────────────────────────────────────┘")
    
    # -----------------------------------------------------------------
    # Step 7: Top Predictive Features
    # -----------------------------------------------------------------
    print("\n TOP PREDICTIVE TOPOLOGICAL FEATURES")
    print("-" * 55)
    
    clf_topo.fit(X_topo, y)
    top_features = clf_topo.get_feature_importance(
        feature_names=feature_names,
        top_n=5
    )
    
    for i, (name, score) in enumerate(top_features, 1):
        bar = "█" * int(score * 40)
        print(f"   {i}. {name:<28} {score:.4f} {bar}")
    


if __name__ == "__main__":
    main()
