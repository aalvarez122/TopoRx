#!/usr/bin/env python
"""
TopoRx Quick Start Example
==========================

A simple example demonstrating the TopoRx pipeline
for predicting cancer drug response using topological
data analysis.

Run with:
    python examples/quick_start.py

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
    print("=" * 60)
    print("🧬 TopoRx: Quick Start Demo")
    print("   Topological Biomarker Discovery for Drug Response")
    print("=" * 60)
    
    # -----------------------------------------------------------------
    # Step 1: Load sample cancer data
    # -----------------------------------------------------------------
    print("\n📊 Loading sample cancer gene expression data...")
    
    from toporx.data import load_sample_data, get_cancer_genes
    
    X, y, gene_names = load_sample_data("default")
    
    print(f"   ✓ Loaded {X.shape[0]} patient samples")
    print(f"   ✓ {X.shape[1]} genes measured")
    print(f"   ✓ {y.sum()} responders, {len(y) - y.sum()} non-responders")
    
    # Show some gene names
    print(f"\n   Sample genes: {', '.join(gene_names[:5])}...")
    
    # -----------------------------------------------------------------
    # Step 2: Compute Persistent Homology
    # -----------------------------------------------------------------
    print("\n🔺 Computing Persistent Homology...")
    
    from toporx.tda import PersistentHomologyComputer
    
    ph = PersistentHomologyComputer(max_dimension=2)
    diagrams = ph.fit_transform(X)
    
    summary = ph.summary()
    for dim, stats in summary.items():
        print(f"   {dim}: {stats['n_features']} features, "
              f"max persistence = {stats['max_persistence']:.3f}")
    
    # -----------------------------------------------------------------
    # Step 3: Extract Topological Features
    # -----------------------------------------------------------------
    print("\n📈 Extracting Topological Features...")
    
    from toporx.tda import TopologicalFeatureExtractor
    
    extractor = TopologicalFeatureExtractor(
        feature_types=["statistics", "entropy", "betti"]
    )
    topo_features = extractor.fit_transform(diagrams)
    feature_names = extractor.get_feature_names()
    
    print(f"   ✓ Extracted {len(topo_features)} topological features")
    print(f"   ✓ Feature types: statistics, entropy, betti curves")
    
    # -----------------------------------------------------------------
    # Step 4: Train Drug Response Classifier
    # -----------------------------------------------------------------
    print("\n🎯 Training Drug Response Classifier...")
    
    from toporx.prediction import DrugResponseClassifier
    
    # Build feature matrix (simplified for demo)
    np.random.seed(42)
    n_samples = X.shape[0]
    feature_matrix = np.array([
        topo_features + np.random.randn(len(topo_features)) * 0.1
        for _ in range(n_samples)
    ])
    
    clf = DrugResponseClassifier(model_type="random_forest")
    cv_results = clf.cross_validate(feature_matrix, y, cv=5)
    
    print(f"   ✓ Model: Random Forest")
    print(f"   ✓ Cross-validation ROC-AUC: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    # -----------------------------------------------------------------
    # Step 5: Compare TDA vs Gene-Based Features
    # -----------------------------------------------------------------
    print("\n📊 Comparing TDA vs Traditional Gene-Based Approach...")
    
    from toporx.prediction.classifier import ComparativeAnalysis
    
    comparison = ComparativeAnalysis()
    results = comparison.compare(
        X_topo=feature_matrix,
        X_genes=X,
        y=y,
        cv=5
    )
    
    print(comparison.summary())
    
    # -----------------------------------------------------------------
    # Step 6: Show Top Predictive Features
    # -----------------------------------------------------------------
    print("\n🏆 Top Predictive Topological Features:")
    
    clf.fit(feature_matrix, y)
    top_features = clf.get_feature_importance(
        feature_names=feature_names,
        top_n=5
    )
    
    for i, (name, score) in enumerate(top_features, 1):
        print(f"   {i}. {name}: {score:.4f}")
    
    # -----------------------------------------------------------------
    # Done!
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print(" Quick start complete!")
    print()
    print("Next steps:")
    print("  • Try the full pipeline: python examples/full_analysis.py")
    print("  • Explore visualizations: python examples/visualization_demo.py")
    print("  • Read the docs: https://github.com/aalvarez122/TopoRx")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
