"""
Drug Response Classifier
========================

Machine learning models for predicting cancer drug response
using topological or gene-based biomarkers.

When used with TopoRx synthetic data, the comparison between
TDA and gene-based features is for DEMONSTRATION only.
The synthetic drug response is engineered to correlate with
specific biomarkers, so results don't reflect real-world performance.

For meaningful comparisons, use real GDSC/CCLE data.

Author: Angelica Alvarez
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)


class DrugResponseClassifier:
    """
    Classifier for predicting drug response from features.
    
    Wraps multiple sklearn algorithms with utilities for
    cross-validation and feature importance analysis.
    
    Parameters
    ----------
    model_type : str, default="random_forest"
        Type of classifier:
        - "random_forest": Good for feature importance, handles non-linear
        - "gradient_boosting": Often best performance, slower
        - "logistic": Interpretable, linear decision boundary
        - "svm": Good for small datasets, less interpretable
    n_estimators : int, default=100
        Number of trees (for ensemble methods)
    max_depth : int, default=10
        Maximum tree depth (prevents overfitting)
    random_state : int, default=42
        Random seed for reproducibility
    scale_features : bool, default=True
        Whether to standardize features (recommended for logistic/svm)
        
    Attributes
    ----------
    model_ : sklearn estimator
        Fitted model
    scaler_ : StandardScaler
        Fitted scaler (if scale_features=True)
    feature_importances_ : np.ndarray
        Feature importance scores (for tree-based models)
    classes_ : np.ndarray
        Class labels
        
    Examples
    --------
    >>> from toporx.prediction import DrugResponseClassifier
    >>> import numpy as np
    >>> 
    >>> # Features (n_samples, n_features)
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randint(0, 2, 100)
    >>> 
    >>> # Train and evaluate
    >>> clf = DrugResponseClassifier(model_type="random_forest")
    >>> results = clf.cross_validate(X, y, cv=5)
    >>> print(f"ROC-AUC: {results['mean_score']:.3f}")
    
    Notes
    -----
    For cross-validation, scaling is done WITHIN each fold using
    sklearn Pipeline to prevent data leakage.
    """
    
    SUPPORTED_MODELS = {
        "random_forest",
        "gradient_boosting", 
        "logistic",
        "svm"
    }
    
    def __init__(
        self,
        model_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
        scale_features: bool = True
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Supported: {self.SUPPORTED_MODELS}"
            )
        
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.scale_features = scale_features
        
        self.model_ = None
        self.scaler_ = None
        self.feature_importances_ = None
        self.classes_ = None
        self._is_fitted = False
    
    def _create_model(self):
        """
        Create the underlying sklearn model.
        
        Model choices:
        - RandomForest: Ensemble of decision trees, good default
        - GradientBoosting: Sequential ensemble, often better but slower
        - LogisticRegression: Linear, interpretable coefficients
        - SVC: Support vectors, good for small datasets
        
        All models use class_weight="balanced" to handle
        imbalanced drug response data.
        """
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced"
            )
        elif self.model_type == "gradient_boosting":
            # Note: GradientBoosting doesn't support class_weight
            # Consider using sample_weight in fit() for imbalanced data
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
        elif self.model_type == "logistic":
            return LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight="balanced"
            )
        elif self.model_type == "svm":
            return SVC(
                random_state=self.random_state,
                probability=True,
                class_weight="balanced"
            )
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> 'DrugResponseClassifier':
        """
        Fit the classifier.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature matrix (topological or gene-based)
        y : np.ndarray of shape (n_samples,)
            Drug response labels (0=resistant, 1=sensitive)
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        
        # Scale features if requested
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X, y)
        
        # Store classes
        self.classes_ = self.model_.classes_
        
        # Extract feature importances
        self._extract_feature_importances()
        
        self._is_fitted = True
        return self
    
    def _extract_feature_importances(self):
        """
        Extract feature importances from fitted model.
        
        Different models provide importances differently:
        - Tree models: feature_importances_ (mean decrease in impurity)
        - Linear models: coef_ (coefficient magnitudes)
        - SVM: Not directly available (would need permutation importance)
        """
        if hasattr(self.model_, 'feature_importances_'):
            # Random Forest, Gradient Boosting
            self.feature_importances_ = self.model_.feature_importances_
        elif hasattr(self.model_, 'coef_'):
            # Logistic Regression
            # For binary classification, coef_ is (1, n_features)
            coef = self.model_.coef_
            if coef.ndim > 1:
                coef = coef[0]  # Take first row for binary
            self.feature_importances_ = np.abs(coef)
        else:
            # SVM (kernel) - no direct feature importance
            self.feature_importances_ = None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict drug response.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels (0=resistant, 1=sensitive)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        X = np.asarray(X)
        
        if self.scale_features and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict response probabilities.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features
            
        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Class probabilities [P(resistant), P(sensitive)]
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        X = np.asarray(X)
        
        if self.scale_features and self.scaler_ is not None:
            X = self.scaler_.transform(X)
        
        return self.model_.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            True labels
            
        Returns
        -------
        float
            Accuracy (proportion correct)
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation on test data.
        
        Parameters
        ----------
        X : np.ndarray
            Test features
        y : np.ndarray
            True labels
            
        Returns
        -------
        dict
            Metrics: accuracy, roc_auc, f1_score, precision, recall
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]
        
        return {
            "accuracy": float(accuracy_score(y, predictions)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "f1_score": float(f1_score(y, predictions)),
            "precision": float(precision_score(y, predictions)),
            "recall": float(recall_score(y, predictions))
        }
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "roc_auc"
    ) -> Dict[str, float]:
        """
        Perform cross-validation with proper data handling.
        
        Uses sklearn Pipeline to ensure scaling is done WITHIN
        each fold, preventing data leakage.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Number of folds
        scoring : str
            Scoring metric: "roc_auc", "accuracy", "f1", etc.
            
        Returns
        -------
        dict
            Cross-validation results including mean, std, and all scores
            
        Notes
        -----
        Data leakage prevention:
        Scaling is performed within each CV fold using Pipeline.
        This ensures the test fold is truly unseen during training.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Create pipeline with scaling inside
        # This prevents data leakage by fitting scaler only on training folds
        if self.scale_features:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', self._create_model())
            ])
        else:
            pipeline = Pipeline([
                ('model', self._create_model())
            ])
        
        # Stratified K-Fold maintains class proportions in each fold
        cv_splitter = StratifiedKFold(
            n_splits=cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scores = cross_val_score(
            pipeline, X, y, 
            cv=cv_splitter, 
            scoring=scoring
        )
        
        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
            "scoring": scoring,
            "cv_folds": cv
        }
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top important features.
        
        Parameters
        ----------
        feature_names : list of str, optional
            Names for each feature. If None, uses "feature_0", etc.
        top_n : int
            Number of top features to return
            
        Returns
        -------
        list of (name, importance) tuples
            Sorted by importance descending
            
        Notes
        -----
        Feature importance interpretation:
        - Random Forest: Mean decrease in impurity (Gini importance)
        - Logistic Regression: Absolute coefficient magnitude
        - SVM: Not available (consider permutation importance instead)
        """
        if self.feature_importances_ is None:
            raise ValueError(
                "Feature importances not available for this model. "
                "Try random_forest or logistic regression."
            )
        
        n_features = len(self.feature_importances_)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if len(feature_names) != n_features:
            raise ValueError(
                f"feature_names length ({len(feature_names)}) "
                f"doesn't match features ({n_features})"
            )
        
        # Pair names with importances and sort
        paired = list(zip(feature_names, self.feature_importances_))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        return paired[:top_n]
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"DrugResponseClassifier("
            f"model_type='{self.model_type}', "
            f"status={status})"
        )


class ComparativeAnalysis:
    """
    Compare different feature sets for drug response prediction.
    
    When using SYNTHETIC data (from toporx.data.load_sample_data),
    this comparison is for DEMONSTRATION ONLY. Any "improvement" shown is artificial.
    
    For meaningful scientific comparisons, use:
    - Real GDSC/CCLE drug response data
    - External validation sets
    - Proper statistical testing
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility
        
    Examples
    --------
    >>> from toporx.prediction import ComparativeAnalysis
    >>> 
    >>> comparison = ComparativeAnalysis()
    >>> results = comparison.compare(
    ...     X_topo=topological_features,
    ...     X_genes=gene_expression,
    ...     y=drug_response,
    ...     cv=5
    ... )
    >>> print(comparison.summary())
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results_ = None
    
    def compare(
        self,
        X_topo: np.ndarray,
        X_genes: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict[str, Dict]:
        """
        Compare topological vs gene-based vs combined features.
        
        Parameters
        ----------
        X_topo : np.ndarray of shape (n_samples, n_topo_features)
            Topological features from TDA
        X_genes : np.ndarray of shape (n_samples, n_genes)
            Gene expression features
        y : np.ndarray of shape (n_samples,)
            Drug response labels
        cv : int
            Cross-validation folds
            
        Returns
        -------
        dict
            Results for each approach (topological, gene_based, combined)
            
        Notes
        -----
        All three approaches use the same:
        - Random Forest classifier
        - Cross-validation splits (same random_state)
        - Feature scaling (within CV folds)
        
        This ensures fair comparison of the feature sets themselves.
        """
        results = {}
        
        # Evaluate topological features
        clf_topo = DrugResponseClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )
        results["topological"] = clf_topo.cross_validate(X_topo, y, cv=cv)
        results["topological"]["n_features"] = X_topo.shape[1]
        
        # Evaluate gene-based features
        clf_genes = DrugResponseClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )
        results["gene_based"] = clf_genes.cross_validate(X_genes, y, cv=cv)
        results["gene_based"]["n_features"] = X_genes.shape[1]
        
        # Evaluate combined features
        X_combined = np.hstack([X_topo, X_genes])
        clf_combined = DrugResponseClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )
        results["combined"] = clf_combined.cross_validate(X_combined, y, cv=cv)
        results["combined"]["n_features"] = X_combined.shape[1]
        
        # Compute improvement metrics
        topo_score = results["topological"]["mean_score"]
        gene_score = results["gene_based"]["mean_score"]
        
        if gene_score > 0:
            relative_improvement = ((topo_score - gene_score) / gene_score) * 100
        else:
            relative_improvement = 0.0
        
        results["comparison"] = {
            "absolute_difference": topo_score - gene_score,
            "relative_percent": relative_improvement,
            "topo_better": topo_score > gene_score,
            "note": "With synthetic data, this comparison is for demonstration only"
        }
        
        self.results_ = results
        return results
    
    def summary(self) -> str:
        """
        Generate text summary of comparison.
        
        Returns
        -------
        str
            Formatted summary table
        """
        if self.results_ is None:
            return "No comparison run yet. Call compare() first."
        
        r = self.results_
        
        lines = [
            "",
            "=" * 60,
            "DRUG RESPONSE PREDICTION COMPARISON",
            "=" * 60,
            "",
            f"  Topological Features ({r['topological']['n_features']} features):",
            f"    ROC-AUC = {r['topological']['mean_score']:.3f} ± {r['topological']['std_score']:.3f}",
            "",
            f"  Gene-Based Features ({r['gene_based']['n_features']} features):",
            f"    ROC-AUC = {r['gene_based']['mean_score']:.3f} ± {r['gene_based']['std_score']:.3f}",
            "",
            f"  Combined Features ({r['combined']['n_features']} features):",
            f"    ROC-AUC = {r['combined']['mean_score']:.3f} ± {r['combined']['std_score']:.3f}",
            "",
            "-" * 60,
            f"  Difference (TDA - Genes): {r['comparison']['absolute_difference']:+.3f}",
            f"  Relative: {r['comparison']['relative_percent']:+.1f}%",
            "",
            "    Note: With synthetic data, comparison is for demonstration.",
            "=" * 60,
            ""
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        status = "completed" if self.results_ is not None else "not run"
        return f"ComparativeAnalysis(status={status})"
