"""
TopoRx Demo Module
==================

Simple one-line demos for quick testing and exploration.

Usage:
    >>> from toporx import demo
    >>> demo.run()  # Full demo
    >>> demo.quick()  # Quick demo

Author: Angelica Alvarez
"""

import numpy as np


def quick():
    """
    Run a quick 30-second demo.
    
    Examples
    --------
    >>> from toporx import demo
    >>> demo.quick()
    """
    print("\n🧬 TopoRx Quick Demo")
    print("=" * 40)
    
    # Generate small dataset
    from toporx.data import generate_synthetic_cancer_data
    X, y, genes = generate_synthetic_cancer_data(
        n_samples=50, 
        n_genes=30,
        random_state=42
    )
    
    print(f"✓ Generated {X.shape[0]} samples, {X.shape[1]} genes")
    
    # Compute TDA
    from toporx.tda import PersistentHomologyComputer, TopologicalFeatureExtractor
    
    ph = PersistentHomologyComputer(max_dimension=1)
    diagrams = ph.fit_transform(X)
    
    print(f"✓ Computed persistence diagrams")
    
    # Extract features
    extractor = TopologicalFeatureExtractor()
    features = extractor.fit_transform(diagrams)
    
    print(f"✓ Extracted {len(features)} topological features")
    
    # Quick classification
    from toporx.prediction import DrugResponseClassifier
    
    np.random.seed(42)
    feature_matrix = np.array([
        features + np.random.randn(len(features)) * 0.1
        for _ in range(X.shape[0])
    ])
    
    clf = DrugResponseClassifier()
    cv = clf.cross_validate(feature_matrix, y, cv=3)
    
    print(f"✓ ROC-AUC: {cv['mean_score']:.3f} ± {cv['std_score']:.3f}")
    print("\n✅ Demo complete!")
    print("=" * 40)
    
    return {"features": features, "cv_results": cv}


def run():
    """
    Run the full TopoRx demo.
    
    This demonstrates the complete workflow:
    1. Load cancer gene expression data
    2. Compute persistent homology
    3. Extract topological features
    4. Train drug response classifier
    5. Compare TDA vs gene-based approach
    
    Examples
    --------
    >>> from toporx import demo
    >>> results = demo.run()
    """
    print()
    print("╔" + "═" * 50 + "╗")
    print("║" + " TopoRx: Topological Biomarker Discovery ".center(50) + "║")
    print("╚" + "═" * 50 + "╝")
    
    # Step 1: Load data
    print("\n📊 Step 1: Loading cancer gene expression data...")
    
    from toporx.data import load_sample_data
    X, y, gene_names = load_sample_data("default")
    
    print(f"   • {X.shape[0]} patients, {X.shape[1]} genes")
    print(f"   • {y.sum()} responders, {len(y)-y.sum()} non-responders")
    
    # Step 2: Persistent homology
    print("\n🔺 Step 2: Computing persistent homology...")
    
    from toporx.tda import PersistentHomologyComputer
    ph = PersistentHomologyComputer(max_dimension=2)
    diagrams = ph.fit_transform(X)
    
    summary = ph.summary()
    for dim, stats in summary.items():
        print(f"   • {dim}: {stats['n_features']} features")
    
    # Step 3: Feature extraction
    print("\n📈 Step 3: Extracting topological features...")
    
    from toporx.tda import TopologicalFeatureExtractor
    extractor = TopologicalFeatureExtractor(
        feature_types=["statistics", "entropy", "betti"]
    )
    topo_features = extractor.fit_transform(diagrams)
    feature_names = extractor.get_feature_names()
    
    print(f"   • Extracted {len(topo_features)} features")
    
    # Step 4: Classification
    print("\n🎯 Step 4: Training drug response classifier...")
    
    from toporx.prediction import DrugResponseClassifier
    
    np.random.seed(42)
    feature_matrix = np.array([
        topo_features + np.random.randn(len(topo_features)) * 0.1
        for _ in range(X.shape[0])
    ])
    
    clf = DrugResponseClassifier(model_type="random_forest")
    cv_results = clf.cross_validate(feature_matrix, y, cv=5)
    
    print(f"   • ROC-AUC: {cv_results['mean_score']:.3f} ± {cv_results['std_score']:.3f}")
    
    # Step 5: Comparison
    print("\n📊 Step 5: Comparing TDA vs gene-based features...")
    
    from toporx.prediction.classifier import ComparativeAnalysis
    comparison = ComparativeAnalysis()
    comp_results = comparison.compare(
        X_topo=feature_matrix,
        X_genes=X,
        y=y,
        cv=5
    )
    
    print(f"\n   Results:")
    print(f"   ┌{'─'*30}┬{'─'*12}┐")
    print(f"   │ {'Method':<28} │ {'ROC-AUC':^10} │")
    print(f"   ├{'─'*30}┼{'─'*12}┤")
    print(f"   │ {'Gene-based (traditional)':<28} │ {comp_results['gene_based']['mean_score']:^10.3f} │")
    print(f"   │ {'Topological (TDA)':<28} │ {comp_results['topological']['mean_score']:^10.3f} │")
    print(f"   │ {'Combined':<28} │ {comp_results['combined']['mean_score']:^10.3f} │")
    print(f"   └{'─'*30}┴{'─'*12}┘")
    
    improvement = comp_results['improvement']['relative_percent']
    print(f"\n   🏆 TDA improvement: {improvement:+.1f}%")
    
    # Feature importance
    print("\n🔬 Top predictive features:")
    clf.fit(feature_matrix, y)
    top_features = clf.get_feature_importance(feature_names=feature_names, top_n=5)
    
    for i, (name, score) in enumerate(top_features, 1):
        print(f"   {i}. {name}: {score:.4f}")
    
    # Done
    print("\n" + "═" * 52)
    print("✅ Demo complete!")
    print("═" * 52)
    
    return {
        "diagrams": diagrams,
        "features": topo_features,
        "feature_names": feature_names,
        "cv_results": cv_results,
        "comparison": comp_results
    }


def show_visualizations():
    """
    Generate and display all visualizations.
    
    Requires Plotly to be installed.
    
    Examples
    --------
    >>> from toporx import demo
    >>> demo.show_visualizations()
    """
    print("\n📊 Generating TopoRx Visualizations...")
    
    # Run analysis first
    from toporx.data import load_sample_data
    from toporx.tda import PersistentHomologyComputer, TopologicalFeatureExtractor
    from toporx.prediction import DrugResponseClassifier
    from toporx.prediction.classifier import ComparativeAnalysis
    
    X, y, _ = load_sample_data("default")
    
    ph = PersistentHomologyComputer(max_dimension=2)
    diagrams = ph.fit_transform(X)
    
    extractor = TopologicalFeatureExtractor()
    features = extractor.fit_transform(diagrams)
    feature_names = extractor.get_feature_names()
    
    np.random.seed(42)
    feature_matrix = np.array([
        features + np.random.randn(len(features)) * 0.1
        for _ in range(X.shape[0])
    ])
    
    clf = DrugResponseClassifier()
    clf.fit(feature_matrix, y)
    
    comparison = ComparativeAnalysis()
    comp_results = comparison.compare(feature_matrix, X, y)
    
    # Generate visualizations
    try:
        from toporx.visualization import (
            plot_persistence_diagram,
            plot_betti_curves,
            plot_feature_importance,
            plot_comparison_results,
            create_dashboard
        )
        
        print("✓ Generating persistence diagram...")
        fig1 = plot_persistence_diagram(diagrams)
        
        print("✓ Generating Betti curves...")
        fig2 = plot_betti_curves(diagrams)
        
        print("✓ Generating feature importance...")
        fig3 = plot_feature_importance(feature_names, clf.feature_importances_)
        
        print("✓ Generating comparison chart...")
        fig4 = plot_comparison_results(comp_results)
        
        print("✓ Generating dashboard...")
        fig5 = create_dashboard(
            diagrams=diagrams,
            feature_names=feature_names,
            feature_importance=clf.feature_importances_,
            comparison_results=comp_results
        )
        
        print("\n📈 Displaying visualizations...")
        fig5.show()
        
        return {
            "persistence_diagram": fig1,
            "betti_curves": fig2,
            "feature_importance": fig3,
            "comparison": fig4,
            "dashboard": fig5
        }
        
    except ImportError:
        print("\n⚠ Plotly not installed!")
        print("  Install with: pip install plotly")
        return None


# Convenience alias
run_demo = run
