"""
Topological Feature Extraction Module
=====================================

Extract machine learning-ready features from persistence diagrams.
These features capture the "shape" of data for drug response prediction.

Author: Angelica Alvarez
"""

import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from scipy.stats import entropy


class TopologicalFeatureExtractor:
    """
    Extract numerical features from persistence diagrams.
    
    Converts topological information into feature vectors suitable
    for machine learning models. Implements multiple feature types:
    
    - Statistical summaries (mean, std, max persistence)
    - Persistence entropy (information content)
    - Betti curve features (topological evolution)
    - Persistence landscape features (functional summaries)
    
    Parameters
    ----------
    feature_types : list of str, default=["statistics", "entropy", "betti"]
        Types of features to extract:
        - "statistics": Basic statistical summaries
        - "entropy": Persistence entropy
        - "betti": Betti curve summaries
        - "landscape": Persistence landscape features
    n_bins : int, default=20
        Number of bins for Betti curve discretization
    landscape_resolution : int, default=50
        Resolution for persistence landscape computation
        
    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features
    n_features_ : int
        Total number of features
        
    Examples
    --------
    >>> from toporx.tda import PersistentHomologyComputer, TopologicalFeatureExtractor
    >>> import numpy as np
    >>> 
    >>> # Compute persistence diagrams
    >>> X = np.random.randn(100, 50)
    >>> ph = PersistentHomologyComputer(max_dimension=1)
    >>> diagrams = ph.fit_transform(X)
    >>> 
    >>> # Extract features
    >>> extractor = TopologicalFeatureExtractor()
    >>> features = extractor.fit_transform(diagrams)
    >>> print(f"Extracted {len(features)} features")
    """
    
    def __init__(
        self,
        feature_types: List[str] = None,
        n_bins: int = 20,
        landscape_resolution: int = 50
    ):
        if feature_types is None:
            feature_types = ["statistics", "entropy", "betti"]
        
        valid_types = {"statistics", "entropy", "betti", "landscape"}
        for ft in feature_types:
            if ft not in valid_types:
                raise ValueError(f"Unknown feature type: {ft}. Valid: {valid_types}")
        
        self.feature_types = feature_types
        self.n_bins = n_bins
        self.landscape_resolution = landscape_resolution
        
        self.feature_names_ = []
        self.n_features_ = 0
        self._is_fitted = False
        self._max_dimension = None
        self._filtration_range = None
    
    def fit(
        self, 
        diagrams: List[np.ndarray], 
        y=None
    ) -> 'TopologicalFeatureExtractor':
        """
        Fit the feature extractor to persistence diagrams.
        
        Parameters
        ----------
        diagrams : list of np.ndarray
            Persistence diagrams for each homology dimension
        y : ignored
            
        Returns
        -------
        self
        """
        self._max_dimension = len(diagrams) - 1
        
        # Determine filtration range
        all_values = []
        for diagram in diagrams:
            if len(diagram) > 0:
                all_values.extend(diagram[:, 0].tolist())
                all_values.extend(diagram[:, 1].tolist())
        
        if all_values:
            self._filtration_range = (min(all_values), max(all_values))
        else:
            self._filtration_range = (0, 1)
        
        # Build feature names
        self.feature_names_ = self._build_feature_names()
        self.n_features_ = len(self.feature_names_)
        
        self._is_fitted = True
        return self
    
    def transform(self, diagrams: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from persistence diagrams.
        
        Parameters
        ----------
        diagrams : list of np.ndarray
            Persistence diagrams
            
        Returns
        -------
        np.ndarray of shape (n_features,)
            Extracted feature vector
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        
        features = []
        
        for dim, diagram in enumerate(diagrams):
            if "statistics" in self.feature_types:
                features.extend(self._extract_statistics(diagram, dim))
            
            if "entropy" in self.feature_types:
                features.extend(self._extract_entropy(diagram, dim))
            
            if "betti" in self.feature_types:
                features.extend(self._extract_betti_features(diagram, dim))
            
            if "landscape" in self.feature_types:
                features.extend(self._extract_landscape_features(diagram, dim))
        
        return np.array(features)
    
    def fit_transform(
        self, 
        diagrams: List[np.ndarray], 
        y=None
    ) -> np.ndarray:
        """
        Fit and extract features in one step.
        
        Parameters
        ----------
        diagrams : list of np.ndarray
            Persistence diagrams
        y : ignored
            
        Returns
        -------
        np.ndarray
            Feature vector
        """
        return self.fit(diagrams, y).transform(diagrams)
    
    def transform_multiple(
        self, 
        diagrams_list: List[List[np.ndarray]]
    ) -> np.ndarray:
        """
        Extract features from multiple sets of persistence diagrams.
        
        Parameters
        ----------
        diagrams_list : list of list of np.ndarray
            List of persistence diagrams for multiple samples
            
        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Feature matrix
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform_multiple()")
        
        return np.array([self.transform(d) for d in diagrams_list])
    
    def _build_feature_names(self) -> List[str]:
        """Build list of feature names."""
        names = []
        
        for dim in range(self._max_dimension + 1):
            prefix = f"H{dim}"
            
            if "statistics" in self.feature_types:
                names.extend([
                    f"{prefix}_n_features",
                    f"{prefix}_mean_persistence",
                    f"{prefix}_std_persistence",
                    f"{prefix}_max_persistence",
                    f"{prefix}_sum_persistence",
                    f"{prefix}_mean_birth",
                    f"{prefix}_mean_death",
                    f"{prefix}_persistence_range"
                ])
            
            if "entropy" in self.feature_types:
                names.extend([
                    f"{prefix}_persistence_entropy",
                    f"{prefix}_normalized_entropy"
                ])
            
            if "betti" in self.feature_types:
                names.extend([
                    f"{prefix}_betti_max",
                    f"{prefix}_betti_mean",
                    f"{prefix}_betti_auc"
                ])
            
            if "landscape" in self.feature_types:
                names.extend([
                    f"{prefix}_landscape_max",
                    f"{prefix}_landscape_mean",
                    f"{prefix}_landscape_auc"
                ])
        
        return names
    
    def _extract_statistics(
        self, 
        diagram: np.ndarray, 
        dim: int
    ) -> List[float]:
        """Extract statistical features from a persistence diagram."""
        if len(diagram) == 0:
            return [0.0] * 8
        
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistence = deaths - births
        
        return [
            float(len(diagram)),                    # n_features
            float(np.mean(persistence)),            # mean_persistence
            float(np.std(persistence)),             # std_persistence
            float(np.max(persistence)),             # max_persistence
            float(np.sum(persistence)),             # sum_persistence
            float(np.mean(births)),                 # mean_birth
            float(np.mean(deaths)),                 # mean_death
            float(np.max(persistence) - np.min(persistence))  # range
        ]
    
    def _extract_entropy(
        self, 
        diagram: np.ndarray, 
        dim: int
    ) -> List[float]:
        """
        Extract persistence entropy features.
        
        Persistence entropy measures the information content
        of the topological features.
        """
        if len(diagram) == 0:
            return [0.0, 0.0]
        
        persistence = diagram[:, 1] - diagram[:, 0]
        persistence = persistence[persistence > 0]
        
        if len(persistence) == 0:
            return [0.0, 0.0]
        
        # Normalize persistence values to form probability distribution
        total = np.sum(persistence)
        if total == 0:
            return [0.0, 0.0]
        
        probabilities = persistence / total
        
        # Compute entropy
        pers_entropy = entropy(probabilities)
        
        # Normalized entropy (0 to 1)
        max_entropy = np.log(len(probabilities)) if len(probabilities) > 1 else 1
        normalized_entropy = pers_entropy / max_entropy if max_entropy > 0 else 0
        
        return [float(pers_entropy), float(normalized_entropy)]
    
    def _extract_betti_features(
        self, 
        diagram: np.ndarray, 
        dim: int
    ) -> List[float]:
        """
        Extract features from Betti curves.
        
        Betti curves show how the number of topological features
        evolves across the filtration.
        """
        if len(diagram) == 0:
            return [0.0, 0.0, 0.0]
        
        # Create filtration values
        filt_min, filt_max = self._filtration_range
        filtration_values = np.linspace(filt_min, filt_max, self.n_bins)
        
        # Compute Betti number at each filtration value
        betti_curve = np.zeros(self.n_bins)
        
        for i, filt in enumerate(filtration_values):
            # Count features alive at this filtration value
            alive = np.sum((diagram[:, 0] <= filt) & (diagram[:, 1] > filt))
            betti_curve[i] = alive
        
        # Extract features from Betti curve
        betti_max = float(np.max(betti_curve))
        betti_mean = float(np.mean(betti_curve))
        betti_auc = float(np.trapz(betti_curve, filtration_values))
        
        return [betti_max, betti_mean, betti_auc]
    
    def _extract_landscape_features(
        self, 
        diagram: np.ndarray, 
        dim: int
    ) -> List[float]:
        """
        Extract features from persistence landscapes.
        
        Persistence landscapes are functional summaries that
        are stable and suitable for statistical analysis.
        """
        if len(diagram) == 0:
            return [0.0, 0.0, 0.0]
        
        # Compute first persistence landscape
        filt_min, filt_max = self._filtration_range
        t_values = np.linspace(filt_min, filt_max, self.landscape_resolution)
        
        landscape = np.zeros(self.landscape_resolution)
        
        for i, t in enumerate(t_values):
            # Tent function values for each point
            tent_values = []
            for birth, death in diagram:
                mid = (birth + death) / 2
                half_life = (death - birth) / 2
                
                if birth <= t <= death:
                    if t <= mid:
                        tent_values.append(t - birth)
                    else:
                        tent_values.append(death - t)
            
            # First landscape is max of tent functions
            landscape[i] = max(tent_values) if tent_values else 0
        
        # Extract features
        landscape_max = float(np.max(landscape))
        landscape_mean = float(np.mean(landscape))
        landscape_auc = float(np.trapz(landscape, t_values))
        
        return [landscape_max, landscape_mean, landscape_auc]
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of all extracted features.
        
        Returns
        -------
        list of str
            Feature names
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        return self.feature_names_.copy()
    
    def get_feature_importance_names(
        self, 
        importance_scores: np.ndarray, 
        top_n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top features by importance score.
        
        Parameters
        ----------
        importance_scores : np.ndarray
            Feature importance scores (e.g., from Random Forest)
        top_n : int
            Number of top features to return
            
        Returns
        -------
        list of (name, score) tuples
            Top features with their importance scores
        """
        if len(importance_scores) != len(self.feature_names_):
            raise ValueError("Importance scores length doesn't match features")
        
        paired = list(zip(self.feature_names_, importance_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        return paired[:top_n]
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TopologicalFeatureExtractor("
            f"feature_types={self.feature_types}, "
            f"n_features={self.n_features_ if self._is_fitted else 'N/A'}, "
            f"status={status})"
        )


def extract_features_quick(
    diagrams: List[np.ndarray],
    feature_types: List[str] = None
) -> np.ndarray:
    """
    Quick function to extract topological features.
    
    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams
    feature_types : list of str, optional
        Types of features to extract
        
    Returns
    -------
    np.ndarray
        Feature vector
        
    Examples
    --------
    >>> features = extract_features_quick(diagrams)
    """
    extractor = TopologicalFeatureExtractor(feature_types=feature_types)
    return extractor.fit_transform(diagrams)
