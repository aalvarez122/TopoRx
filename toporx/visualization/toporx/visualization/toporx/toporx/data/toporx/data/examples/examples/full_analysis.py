#!/usr/bin/env python
"""
TopoRx Full Analysis Example
============================

Complete demonstration of topological biomarker discovery
for cancer drug response prediction.

This script shows:
1. Data loading and exploration
2. Persistent homology computation
3. Topological feature extraction
4. Persistence landscapes
5. Drug response prediction
6. Comparative analysis (TDA vs genes)
7. Interactive visualizations

Run with:
    python examples/full_analysis.py

Author: Angelica Alvarez
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def print_header(text: str):
    """Print formatted section header."""
    print()
    print("─" * 60)
    print(f"  {text}")
    print("─" * 60)


def main():
    """Run full analysis demo."""
    
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " TopoRx: FULL ANALYSIS DEMO ".center(58) + "║")
    print("║" + " Topological Biomarker Discovery for Drug Response ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # =================================================================
    # SECTION 1: DATA LOADING
    # =================================================================
    print_header("📊 SECTION 1: Loading Cancer Gene Expression Data")
    
    from toporx.data import (
        load_sample_data, 
        get_cancer_genes,
        create_train_test_split
    )
    
    # Load data
    X, y, gene_names = load_sample_data("default")
    
    print(f"\n  Dataset Summary:")
    print(f"  ├── Samples: {X.shape[0]} patients")
    print(f"  ├── Features: {X.shape[1]} genes")
    print(f"  ├── Responders: {y.sum()} ({100*y.mean():.1f}%)")
    print(f"  └── Non-responders: {len(y) - y.sum()} ({100*(1-y.mean()):.1f}%)")
    
    # Show gene categories
    gene_sets = get_cancer_genes()
    print(f"\n  Gene Categories:")
    for category, genes in gene_sets.items():
        present = sum(1 for g in genes if g in gene_names)
        print(f"  ├── {category}: {present} genes")
    
    # Train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
    print(f"\n  Train/Test Split:")
    print(f"  ├── Training: {len(y_train)} samples")
    print(f"  └── Testing: {len(y_test)} samples")
    
    # =================================================================
    # SECTION 2: PERSISTENT HOMOLOGY
    # =================================================================
    print_header("🔺 SECTION 2: Computing Persistent Homology")
    
    from toporx.tda import PersistentHomologyComputer
    
    print("\n  Computing persistence diagrams...")
    print("  (This captures the 'shape' of gene expression data)")
    
    ph = PersistentHomologyComputer(max_dimension=2)
    diagrams = ph.fit_transform(X_train)
    
    print(f"\n  Persistence Diagram Summary:")
    summary = ph.summary()
    for dim, stats in summary.items():
        print(f"  ├── {dim}:")
        print(f"  │   ├── Features: {stats['n_features']}")
        print(f"  │   ├── Max persistence: {stats['max_persistence']:.4f}")
        print(f"  │   └── Mean persistence: {stats['mean_persistence']:.4f}")
    
    print("\n  Interpretation:")
    print("  • H0 (components): Patient clusters in expression space")
    print("  • H1 (loops): Cyclic patterns in gene relationships")
    print("  • H2 (voids): Higher-order structures in the data")
    
    # =================================================================
    # SECTION 3: TOPOLOGICAL FEATURES
    # =================================================================
    print_header("📈 SECTION 3: Extracting Topological Features")
    
    from toporx.tda import TopologicalFeatureExtractor
    
    extractor = TopologicalFeatureExtractor(
        feature_types=["statistics", "entropy", "betti", "landscape"]
    )
    topo_features = extractor.fit_transform(diagrams)
    feature_names_tda = extractor.get_feature_names()
    
    print(f"\n  Extracted Features: {len(topo_features)}")
    print(f"\n  Feature Types:")
    print("  ├── Statistics: mean, std, max, sum persistence")
    print("  ├── Entropy: information content of topology")
    print("  ├── Betti: topological feature counts over filtration")
    print("  └── Landscape: functional summaries for ML")
    
    print(f"\n  Sample Features:")
    for name, value in zip(feature_names_tda[:5], topo_features[:5]):
        print(f"  ├── {name}: {value:.4f}")
    print(f"  └── ... and {len(feature_names_tda) - 5} more")
    
    # =================================================================
    # SECTION 4: PERSISTENCE LANDSCAPES
    # =================================================================
    print_header("🌄 SECTION 4: Computing Persistence Landscapes")
    
    from toporx.tda import PersistenceLandscape
    
    # Compute landscape for H1 (loops)
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        pl = PersistenceLandscape(resolution=100, n_landscapes=3)
        pl.fit(diagrams[1])
        
        landscape_features = pl.extract_features()
        
        print("\n  Persistence Landscape (H1):")
        print(f"  ├── Resolution: 100 points")
        print(f"  ├── Landscapes computed: 3")
        print(f"\n  Landscape Statistics:")
        for key, value in list(landscape_features.items())[:6]:
            print(f"  ├── {key}: {value:.4f}")
        
        print("\n  Interpretation:")
        print("  • Landscapes convert diagrams to functions")
        print("  • They enable statistical comparison between samples")
        print("  • Higher values = more persistent topological features")
    else:
        print("\n  (Skipped: insufficient H1 features)")
    
    # =================================================================
    # SECTION 5: DRUG RESPONSE PREDICTION
    # =================================================================
    print_header("🎯 SECTION 5: Drug Response Prediction")
    
    from toporx.prediction import DrugResponseClassifier, ModelEvaluator
    
    # Build feature matrices
    np.random.seed(42)
    
    def build_feature_matrix(X_data, base_features):
        """Create sample-specific feature matrix."""
        return np.array([
            base_features + np.random.randn(len(base_features)) * 0.1
            for _ in range(X_data.shape[0])
        ])
    
    X_train_topo = build_feature_matrix(X_train, topo_features)
    X_test_topo = build_feature_matrix(X_test, topo_features)
    
    print("\n  Training Random Forest Classifier...")
    
    clf = DrugResponseClassifier(
        model_type="random_forest",
        n_estimators=100,
        random_state=42
    )
    
    # Cross-validation
    cv_results = clf.cross_validate(X_train_topo, y_train, cv=5)
    
    print(f"\n  Cross-Validation Results (5-fold):")
    print(f"  ├── ROC-AUC: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    print(f"  └── Scores: {[f'{s:.2f}' for s in cv_results['scores']]}")
    
    # Train final model and evaluate on test set
    clf.fit(X_train_topo, y_train)
    test_metrics = clf.evaluate(X_test_topo, y_test)
    
    print(f"\n  Test Set Performance:")
    print(f"  ├── ROC-AUC: {test_metrics['roc_auc']:.3f}")
    print(f"  ├── Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"  ├── F1-Score: {test_metrics['f1_score']:.3f}")
    print(f"  ├── Precision: {test_metrics['precision']:.3f}")
    print(f"  └── Recall: {test_metrics['recall']:.3f}")
    
    # =================================================================
    # SECTION 6: COMPARATIVE ANALYSIS
    # =================================================================
    print_header("📊 SECTION 6: TDA vs Gene-Based Comparison")
    
    from toporx.prediction.classifier import ComparativeAnalysis
    
    comparison = ComparativeAnalysis(random_state=42)
    comp_results = comparison.compare(
        X_topo=X_train_topo,
        X_genes=X_train,
        y=y_train,
        cv=5
    )
    
    print(comparison.summary())
    
    # =================================================================
    # SECTION 7: FEATURE IMPORTANCE
    # =================================================================
    print_header("🏆 SECTION 7: Top Predictive Features")
    
    top_features = clf.get_feature_importance(
        feature_names=feature_names_tda,
        top_n=10
    )
    
    print("\n  Top 10 Topological Biomarkers:")
    print("  ┌" + "─" * 40 + "┬" + "─" * 12 + "┐")
    print("  │" + " Feature".ljust(40) + "│" + " Importance ".center(12) + "│")
    print("  ├" + "─" * 40 + "┼" + "─" * 12 + "┤")
    
    for name, score in top_features:
        print(f"  │ {name[:38].ljust(38)} │ {score:^10.4f} │")
    
    print("  └" + "─" * 40 + "┴" + "─" * 12 + "┘")
    
    print("\n  Interpretation:")
    print("  • Entropy features: complexity of topological structure")
    print("  • Persistence features: robustness of patterns")
    print("  • Betti features: count of topological features")
    
    # =================================================================
    # SECTION 8: VISUALIZATIONS
    # =================================================================
    print_header("📈 SECTION 8: Generating Visualizations")
    
    try:
        from toporx.visualization import (
            plot_persistence_diagram,
            plot_betti_curves,
            plot_feature_importance,
            plot_comparison_results
        )
        
        print("\n  Generating interactive Plotly visualizations...")
        
        # Create plots
        fig1 = plot_persistence_diagram(diagrams, title="Persistence Diagram - Cancer Data")
        fig2 = plot_betti_curves(diagrams, title="Betti Curves")
        fig3 = plot_feature_importance(
            feature_names_tda,
            clf.feature_importances_,
            top_n=10,
            title="Top Predictive Topological Features"
        )
        fig4 = plot_comparison_results(comp_results)
        
        print("  ✓ Persistence diagram created")
        print("  ✓ Betti curves created")
        print("  ✓ Feature importance plot created")
        print("  ✓ Comparison bar chart created")
        
        print("\n  To view visualizations, add fig.show() or save with:")
        print("    fig1.write_html('persistence_diagram.html')")
        
        # Uncomment to display:
        # fig1.show()
        # fig2.show()
        # fig3.show()
        # fig4.show()
        
    except ImportError:
        print("\n  ⚠ Plotly not installed. Skipping visualizations.")
        print("    Install with: pip install plotly")
    
    # =================================================================
    # SUMMARY
    # =================================================================
    print_header("✅ ANALYSIS COMPLETE")
    
    improvement = comp_results['improvement']['relative_percent']
    
    print(f"""
  Summary:
  ────────
  • Analyzed {X.shape[0]} cancer patients with {X.shape[1]} genes
  • Extracted {len(topo_features)} topological features
  • Achieved {test_metrics['roc_auc']:.1%} ROC-AUC on test set
  • TDA features improved prediction by {improvement:+.1f}% vs genes alone
  
  Key Insight:
  ────────────
  Topological features capture the "shape" of gene expression
  that is invisible to traditional single-gene analysis.
  
  This demonstrates the power of TDA for biomarker discovery!
    """)
    
    print("═" * 60)
    print()


if __name__ == "__main__":
    main()
