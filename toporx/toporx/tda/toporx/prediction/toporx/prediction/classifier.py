"""
Drug Response Classifier
========================

Machine learning models for predicting cancer drug response
using topological biomarkers.

Author: Angelica Alvarez
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)
import warnings
warnings.filterwarnings('ignore')


class DrugResponseClassifier:
    """
    Classifier for predicting drug response from topological features.
    
    This classifier wraps multiple ML algorithms and provides
    utilities for comparing topological features vs traditional
    gene-based approaches.
    
    Parameters
    ----------
    model_type : str, default="random_forest"
        Type of classifier:
        - "random_forest": Random Forest (default, good for feature importance)
        - "gradient_boosting": Gradient Boosting (often best performance)
        - "logistic": Logistic Regression (interpretable)
        - "svm": Support Vector Machine (good for small datasets)
    n_estimators : int, default=100
        Number of trees (for ensemble methods)
    max_depth : int, default=10
        Maximum tree depth
    random_state : int, default=42
        Random seed for reproducibility
    scale_features : bool, default=True
        Whether to standardize features
        
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
    >>> # Topological features (n_samples, n_features)
    >>> X = np.random.randn(100, 20)
    >>> y = np.random.randint(0, 2, 100)  # 0=non-responder, 1=responder
    >>> 
    >>> # Train classifier
    >>> clf = DrugResponseClassifier(model_type="random_forest")
    >>> clf.fit(X, y)
    >>> 
    >>> # Predict
    >>> predictions = clf.predict(X_new)
    >>> probabilities = clf.predict_proba(X_new)
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
        """Create the underlying sklearn model."""
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced"
            )
        elif self.model_type == "gradient_boosting":
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
            Topological feature matrix
        y : np.ndarray of shape (n_samples,)
            Drug response labels (0=non-responder, 1=responder)
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.ndim}D")
        
        # Scale features
        if self.scale_features:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        
        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X, y)
        
        # Store classes
        self.classes_ = self.model_.classes_
        
        # Extract feature importances (if available)
        if hasattr(self.model_, 'feature_importances_'):
            self.feature_importances_ = self.model_.feature_importances_
        elif hasattr(self.model_, 'coef_'):
            self.feature_importances_ = np.abs(self.model_.coef_).flatten()
        else:
            self.feature_importances_ = None
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict drug response.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Topological features
            
        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted labels
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
            Topological features
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities
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
            Accuracy score
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
            Dictionary of metrics
        """
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)[:, 1]
        
        return {
            "accuracy": accuracy_score(y, predictions),
            "roc_auc": roc_auc_score(y, probabilities),
            "f1_score": f1_score(y, predictions),
            "precision": precision_score(y, predictions),
            "recall": recall_score(y, predictions)
        }
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "roc_auc"
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels
        cv : int
            Number of folds
        scoring : str
            Scoring metric
            
        Returns
        -------
        dict
            Cross-validation results
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.scale_features:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        model = self._create_model()
        
        cv_splitter = StratifiedKFold(
            n_splits=cv, 
            shuffle=True, 
            random_state=self.random_state
        )
        
        scores = cross_val_score(
            model, X, y, 
            cv=cv_splitter, 
            scoring=scoring
        )
        
        return {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": scores.tolist(),
            "scoring": scoring
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
            Names for each feature
        top_n : int
            Number of top features to return
            
        Returns
        -------
        list of (name, importance) tuples
        """
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available for this model")
        
        n_features = len(self.feature_importances_)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if len(feature_names) != n_features:
            raise ValueError("feature_names length doesn't match")
        
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
    Compare topological features vs traditional gene-based features.
    
    This class helps demonstrate the advantage of TDA-based
    biomarkers over traditional approaches.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed
        
    Examples
    --------
    >>> from toporx.prediction import ComparativeAnalysis
    >>> 
    >>> # Compare TDA vs gene-based features
    >>> comparison = ComparativeAnalysis()
    >>> results = comparison.compare(
    ...     X_topo=topological_features,
    ...     X_genes=gene_expression,
    ...     y=drug_response
    ... )
    >>> print(results)
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
        Compare topological vs gene-based features.
        
        Parameters
        ----------
        X_topo : np.ndarray
            Topological features
        X_genes : np.ndarray
            Gene expression features
        y : np.ndarray
            Drug response labels
        cv : int
            Cross-validation folds
            
        Returns
        -------
        dict
            Comparison results for each approach
        """
        results = {}
        
        # Evaluate topological features
        clf_topo = DrugResponseClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )
        results["topological"] = clf_topo.cross_validate(X_topo, y, cv=cv)
        
        # Evaluate gene-based features
        clf_genes = DrugResponseClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )
        results["gene_based"] = clf_genes.cross_validate(X_genes, y, cv=cv)
        
        # Evaluate combined features
        X_combined = np.hstack([X_topo, X_genes])
        clf_combined = DrugResponseClassifier(
            model_type="random_forest",
            random_state=self.random_state
        )
        results["combined"] = clf_combined.cross_validate(X_combined, y, cv=cv)
        
        # Compute improvement
        topo_score = results["topological"]["mean_score"]
        gene_score = results["gene_based"]["mean_score"]
        
        results["improvement"] = {
            "absolute": topo_score - gene_score,
            "relative_percent": ((topo_score - gene_score) / gene_score) * 100
        }
        
        self.results_ = results
        return results
    
    def summary(self) -> str:
        """Generate text summary of comparison."""
        if self.results_ is None:
            return "No comparison run yet. Call compare() first."
        
        r = self.results_
        
        lines = [
            "=" * 50,
            "DRUG RESPONSE PREDICTION COMPARISON",
            "=" * 50,
            "",
            f"Topological Features:  ROC-AUC = {r['topological']['mean_score']:.3f} ± {r['topological']['std_score']:.3f}",
            f"Gene-Based Features:   ROC-AUC = {r['gene_based']['mean_score']:.3f} ± {r['gene_based']['std_score']:.3f}",
            f"Combined Features:     ROC-AUC = {r['combined']['mean_score']:.3f} ± {r['combined']['std_score']:.3f}",
            "",
            f"Improvement: {r['improvement']['relative_percent']:+.1f}%",
            "=" * 50
        ]
        
        return "\n".join(lines)
