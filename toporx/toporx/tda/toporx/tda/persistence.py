"""
Persistent Homology with Ripser
===============================

Computes topological features from gene expression data using
Vietoris-Rips persistent homology 

Author: Angelica Alvarez
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import List, Dict, Optional

try:
    import ripser
except ImportError:
    raise ImportError(
        "Ripser is required for persistent homology.\n"
        "Install with: pip install ripser"
    )


class PersistentHomologyComputer:
    """
    Compute persistent homology using Ripser.
    
    This class computes topological features (H0, H1, H2) from 
    high-dimensional data using Vietoris-Rips filtration.
    
    Parameters
    ----------
    max_dimension : int, default=2
        Maximum homology dimension to compute:
        - H0: Connected components (patient clusters)
        - H1: Loops (cyclic patterns in gene relationships)
        - H2: Voids (higher-order structures)
    metric : str, default="correlation"
        Distance metric. "correlation" is recommended for gene expression
        as it captures expression pattern similarity regardless of magnitude.
    threshold : float, default=np.inf
        Maximum filtration value. Use to limit computation on large datasets.
        
    Attributes
    ----------
    persistence_diagrams_ : list of np.ndarray
        Persistence diagrams for each dimension. Each diagram has shape (n, 2)
        with columns [birth, death].
    distance_matrix_ : np.ndarray
        Computed pairwise distance matrix.
    cocycles_ : list
        Representative cocycles (if computed).
        
    Examples
    --------
    >>> from toporx.tda import PersistentHomologyComputer
    >>> import numpy as np
    >>> 
    >>> # Gene expression: 100 patients × 500 genes
    >>> X = np.random.randn(100, 500)
    >>> 
    >>> # Compute persistent homology
    >>> ph = PersistentHomologyComputer(max_dimension=2)
    >>> diagrams = ph.fit_transform(X)
    >>> 
    >>> # View summary
    >>> print(ph.summary())
    """
    
    def __init__(
        self,
        max_dimension: int = 2,
        metric: str = "correlation",
        threshold: float = np.inf
    ):
        self.max_dimension = max_dimension
        self.metric = metric
        self.threshold = threshold
        
        # Results (populated after fit)
        self.persistence_diagrams_ = None
        self.distance_matrix_ = None
        self.cocycles_ = None
        self._ripser_result = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray, y=None) -> 'PersistentHomologyComputer':
        """
        Compute persistent homology from data.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data matrix. For drug response prediction,
            rows are patients and columns are genes.
        y : ignored
            Present for sklearn API compatibility.
            
        Returns
        -------
        self
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D")
        
        n_samples, n_features = X.shape
        print(f"   Computing persistence for {n_samples} samples, {n_features} features...")
        
        # Step 1: Compute distance matrix
        print(f"   → Distance metric: {self.metric}")
        self.distance_matrix_ = self._compute_distance_matrix(X)
        
        # Step 2: Run Ripser
        print(f"   → Running Ripser (max_dim={self.max_dimension})...")
        self._ripser_result = ripser.ripser(
            self.distance_matrix_,
            maxdim=self.max_dimension,
            thresh=self.threshold,
            distance_matrix=True,
            do_cocycles=True
        )
        
        # Step 3: Extract persistence diagrams
        self.persistence_diagrams_ = self._extract_diagrams()
        self.cocycles_ = self._ripser_result.get('cocycles', [])
        
        self._is_fitted = True
        return self
    
    def _compute_distance_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        For gene expression data, correlation distance is preferred
        because it measures similarity of expression patterns
        regardless of absolute expression levels.
        """
        if self.metric == "correlation":
            # Correlation distance: d(x,y) = 1 - corr(x,y)
            # Range: [0, 2] where 0 = identical, 2 = opposite
            distances = pdist(X, metric='correlation')
        else:
            distances = pdist(X, metric=self.metric)
        
        # Handle NaN values (can occur with constant rows)
        distances = np.nan_to_num(distances, nan=1.0)
        
        return squareform(distances)
    
    def _extract_diagrams(self) -> List[np.ndarray]:
        """
        Extract persistence diagrams from Ripser output.
        
        Removes points with infinite death time (features that never die)
        for cleaner downstream processing.
        """
        diagrams = []
        
        for dim in range(self.max_dimension + 1):
            if dim < len(self._ripser_result['dgms']):
                dgm = self._ripser_result['dgms'][dim].copy()
                
                # Separate finite and infinite points
                finite_mask = np.isfinite(dgm[:, 1])
                finite_dgm = dgm[finite_mask]
                
                # Count infinite points (essential features)
                n_infinite = np.sum(~finite_mask)
                if n_infinite > 0:
                    print(f"   → H{dim}: {len(finite_dgm)} finite + {n_infinite} essential features")
                else:
                    print(f"   → H{dim}: {len(finite_dgm)} features")
                
                diagrams.append(finite_dgm)
            else:
                diagrams.append(np.array([]).reshape(0, 2))
        
        return diagrams
    
    def transform(self, X: np.ndarray = None) -> List[np.ndarray]:
        """
        Return computed persistence diagrams.
        
        Parameters
        ----------
        X : ignored
            Not used after fitting.
            
        Returns
        -------
        list of np.ndarray
            Persistence diagrams for each homology dimension.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        return self.persistence_diagrams_
    
    def fit_transform(self, X: np.ndarray, y=None) -> List[np.ndarray]:
        """
        Compute persistent homology and return diagrams.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data matrix.
            
        Returns
        -------
        list of np.ndarray
            Persistence diagrams for each dimension.
        """
        return self.fit(X, y).transform()
    
    def get_diagram(self, dimension: int = 0) -> np.ndarray:
        """
        Get persistence diagram for a specific dimension.
        
        Parameters
        ----------
        dimension : int
            Homology dimension (0, 1, or 2).
            
        Returns
        -------
        np.ndarray of shape (n_features, 2)
            Persistence diagram with [birth, death] columns.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        if dimension > self.max_dimension:
            raise ValueError(
                f"Dimension {dimension} > max_dimension {self.max_dimension}"
            )
        
        return self.persistence_diagrams_[dimension].copy()
    
    def get_persistence(self, dimension: int = 0) -> np.ndarray:
        """
        Get persistence values (death - birth) for a dimension.
        
        Persistence measures how "significant" each topological
        feature is. Longer persistence = more robust feature.
        
        Parameters
        ----------
        dimension : int
            Homology dimension.
            
        Returns
        -------
        np.ndarray
            Persistence values sorted in descending order.
        """
        dgm = self.get_diagram(dimension)
        
        if len(dgm) == 0:
            return np.array([])
        
        persistence = dgm[:, 1] - dgm[:, 0]
        return np.sort(persistence)[::-1]  # Descending order
    
    def summary(self) -> Dict[str, Dict]:
        """
        Get summary statistics of topological features.
        
        Returns
        -------
        dict
            Summary for each homology dimension including:
            - n_features: Number of topological features
            - max_persistence: Strongest feature
            - mean_persistence: Average feature strength
            - total_persistence: Sum of all persistence values
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        summary = {}
        
        for dim, dgm in enumerate(self.persistence_diagrams_):
            if len(dgm) == 0:
                summary[f"H{dim}"] = {
                    "n_features": 0,
                    "max_persistence": 0.0,
                    "mean_persistence": 0.0,
                    "total_persistence": 0.0,
                    "description": self._get_dim_description(dim)
                }
            else:
                persistence = dgm[:, 1] - dgm[:, 0]
                summary[f"H{dim}"] = {
                    "n_features": len(dgm),
                    "max_persistence": float(np.max(persistence)),
                    "mean_persistence": float(np.mean(persistence)),
                    "total_persistence": float(np.sum(persistence)),
                    "description": self._get_dim_description(dim)
                }
        
        return summary
    
    def _get_dim_description(self, dim: int) -> str:
        """Get biological interpretation for each dimension."""
        descriptions = {
            0: "Connected components (patient subgroups)",
            1: "Loops (cyclic gene relationships)",
            2: "Voids (higher-order structures)"
        }
        return descriptions.get(dim, f"H{dim} features")
    
    def get_most_persistent_features(
        self, 
        dimension: int = 1, 
        top_n: int = 5
    ) -> np.ndarray:
        """
        Get the most persistent (significant) features.
        
        Parameters
        ----------
        dimension : int
            Homology dimension.
        top_n : int
            Number of top features to return.
            
        Returns
        -------
        np.ndarray of shape (top_n, 2)
            Top persistence pairs [birth, death].
        """
        dgm = self.get_diagram(dimension)
        
        if len(dgm) == 0:
            return np.array([]).reshape(0, 2)
        
        # Sort by persistence (death - birth)
        persistence = dgm[:, 1] - dgm[:, 0]
        top_indices = np.argsort(persistence)[::-1][:top_n]
        
        return dgm[top_indices]
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"PersistentHomologyComputer("
            f"max_dimension={self.max_dimension}, "
            f"metric='{self.metric}', "
            f"status={status})"
        )


def compute_persistence_quick(
    X: np.ndarray,
    max_dimension: int = 1,
    metric: str = "correlation"
) -> List[np.ndarray]:
    """
    Quick function to compute persistence diagrams.
    
    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    max_dimension : int
        Maximum homology dimension.
    metric : str
        Distance metric.
        
    Returns
    -------
    list of np.ndarray
        Persistence diagrams.
        
    Examples
    --------
    >>> diagrams = compute_persistence_quick(X, max_dimension=1)
    >>> h0, h1 = diagrams[0], diagrams[1]
    """
    ph = PersistentHomologyComputer(
        max_dimension=max_dimension,
        metric=metric
    )
    return ph.fit_transform(X)
