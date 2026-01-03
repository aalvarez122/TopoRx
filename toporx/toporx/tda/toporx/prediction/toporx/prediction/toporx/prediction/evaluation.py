"""
Model Evaluation Module
=======================

Comprehensive evaluation metrics and utilities for
drug response prediction models.

Author: Angelica Alvarez
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, learning_curve
)


class ModelEvaluator:
    """
    Comprehensive model evaluation for drug response prediction.
    
    Provides detailed metrics, statistical tests, and
    visualization-ready outputs.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility
        
    Examples
    --------
    >>> from toporx.prediction import ModelEvaluator
    >>> 
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.evaluate(y_true, y_pred, y_prob)
    >>> print(evaluator.summary())
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.metrics_ = None
        self.curves_ = None
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_prob : np.ndarray, optional
            Predicted probabilities for positive class
            
        Returns
        -------
        dict
            Dictionary of evaluation metrics
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "specificity": self._compute_specificity(y_true, y_pred)
        }
        
        # Compute AUC metrics if probabilities provided
        if y_prob is not None:
            y_prob = np.asarray(y_prob)
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
            metrics["avg_precision"] = average_precision_score(y_true, y_prob)
            
            # Store curves for plotting
            self.curves_ = self._compute_curves(y_true, y_prob)
        
        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_positives"] = int(tp)
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
        
        self.metrics_ = metrics
        return metrics
    
    def _compute_specificity(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Compute specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, _, _ = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _compute_curves(
        self, 
        y_true: np.ndarray, 
        y_prob: np.ndarray
    ) -> Dict:
        """Compute ROC and PR curves."""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
        
        return {
            "roc": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "thresholds": roc_thresholds.tolist()
            },
            "pr": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist()
            }
        }
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ROC curve data for plotting.
        
        Returns
        -------
        tuple of (fpr, tpr)
        """
        if self.curves_ is None:
            raise RuntimeError("Must call evaluate() with y_prob first")
        
        return (
            np.array(self.curves_["roc"]["fpr"]),
            np.array(self.curves_["roc"]["tpr"])
        )
    
    def get_pr_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Precision-Recall curve data.
        
        Returns
        -------
        tuple of (recall, precision)
        """
        if self.curves_ is None:
            raise RuntimeError("Must call evaluate() with y_prob first")
        
        return (
            np.array(self.curves_["pr"]["recall"]),
            np.array(self.curves_["pr"]["precision"])
        )
    
    def summary(self) -> str:
        """
        Generate text summary of evaluation.
        
        Returns
        -------
        str
            Formatted summary
        """
        if self.metrics_ is None:
            return "No evaluation run yet. Call evaluate() first."
        
        m = self.metrics_
        
        lines = [
            "┌" + "─" * 40 + "┐",
            "│" + " DRUG RESPONSE PREDICTION METRICS ".center(40) + "│",
            "├" + "─" * 40 + "┤",
        ]
        
        # Main metrics
        main_metrics = [
            ("Accuracy", "accuracy"),
            ("ROC-AUC", "roc_auc"),
            ("F1-Score", "f1_score"),
            ("Precision", "precision"),
            ("Recall (Sensitivity)", "recall"),
            ("Specificity", "specificity")
        ]
        
        for name, key in main_metrics:
            if key in m:
                lines.append(f"│ {name:<25} {m[key]:>10.3f}  │")
        
        lines.append("├" + "─" * 40 + "┤")
        
        # Confusion matrix
        if "true_positives" in m:
            lines.append("│" + " Confusion Matrix ".center(40) + "│")
            lines.append(f"│ True Positives:  {m['true_positives']:<5}  True Negatives:  {m['true_negatives']:<5} │")
            lines.append(f"│ False Positives: {m['false_positives']:<5}  False Negatives: {m['false_negatives']:<5} │")
        
        lines.append("└" + "─" * 40 + "┘")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Return metrics as dictionary."""
        if self.metrics_ is None:
            return {}
        return self.metrics_.copy()


def evaluate_cross_validation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Perform cross-validation with multiple metrics.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Number of folds
    scoring : list of str
        Metrics to compute
        
    Returns
    -------
    dict
        Results for each metric
    """
    if scoring is None:
        scoring = ["accuracy", "roc_auc", "f1"]
    
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    results = {}
    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=metric)
        results[metric] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores.tolist()
        }
    
    return results


def compute_learning_curve(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    n_points: int = 10
) -> Dict[str, np.ndarray]:
    """
    Compute learning curve data.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    cv : int
        Cross-validation folds
    n_points : int
        Number of training set sizes
        
    Returns
    -------
    dict
        Learning curve data
    """
    train_sizes = np.linspace(0.1, 1.0, n_points)
    
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=train_sizes,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    return {
        "train_sizes": train_sizes_abs,
        "train_mean": np.mean(train_scores, axis=1),
        "train_std": np.std(train_scores, axis=1),
        "val_mean": np.mean(val_scores, axis=1),
        "val_std": np.std(val_scores, axis=1)
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_func=roc_auc_score,
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_prob : np.ndarray
        Predicted probabilities
    metric_func : callable
        Metric function
    n_bootstrap : int
        Number of bootstrap samples
    confidence : float
        Confidence level
        
    Returns
    -------
    tuple of (mean, lower, upper)
    """
    np.random.seed(42)
    n_samples = len(y_true)
    
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, n_samples, n_samples)
        score = metric_func(y_true[indices], y_prob[indices])
        scores.append(score)
    
    scores = np.array(scores)
    
    alpha = (1 - confidence) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)
    
    return float(np.mean(scores)), float(lower), float(upper)
