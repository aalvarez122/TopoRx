"""
Persistence Landscape Module
============================

Persistence landscapes are functional summaries of persistence diagrams
that enable statistical analysis and machine learning on topological features.

Reference:
    Bubenik, P. (2015). Statistical topological data analysis using 
    persistence landscapes. JMLR.

Author: Angelica Alvarez
"""

import numpy as np
from typing import List, Optional, Tuple, Union


class PersistenceLandscape:
    """
    Compute and manipulate persistence landscapes.
    
    Persistence landscapes transform persistence diagrams into
    functional representations that can be averaged, compared,
    and used in machine learning pipelines.
    
    Parameters
    ----------
    resolution : int, default=100
        Number of points to sample the landscape
    n_landscapes : int, default=5
        Number of landscape functions to compute (λ₁, λ₂, ..., λₙ)
    
    Attributes
    ----------
    landscapes_ : np.ndarray
        Computed landscape functions of shape (n_landscapes, resolution)
    t_values_ : np.ndarray
        Filtration values where landscape is evaluated
    
    Examples
    --------
    >>> from toporx.tda import PersistenceLandscape
    >>> import numpy as np
    >>> 
    >>> # Create persistence diagram
    >>> diagram = np.array([[0.1, 0.5], [0.2, 0.8], [0.3, 0.6]])
    >>> 
    >>> # Compute landscape
    >>> pl = PersistenceLandscape(resolution=100, n_landscapes=3)
    >>> pl.fit(diagram)
    >>> 
    >>> # Get landscape values
    >>> landscapes = pl.get_landscapes()
    """
    
    def __init__(
        self,
        resolution: int = 100,
        n_landscapes: int = 5
    ):
        self.resolution = resolution
        self.n_landscapes = n_landscapes
        
        self.landscapes_ = None
        self.t_values_ = None
        self._diagram = None
        self._is_fitted = False
    
    def fit(
        self, 
        diagram: np.ndarray,
        t_range: Optional[Tuple[float, float]] = None
    ) -> 'PersistenceLandscape':
        """
        Compute persistence landscape from a persistence diagram.
        
        Parameters
        ----------
        diagram : np.ndarray of shape (n_points, 2)
            Persistence diagram with (birth, death) pairs
        t_range : tuple of (min, max), optional
            Range of filtration values. If None, inferred from data.
            
        Returns
        -------
        self
        """
        diagram = np.asarray(diagram)
        
        if diagram.ndim != 2 or diagram.shape[1] != 2:
            if diagram.size == 0:
                diagram = np.array([]).reshape(0, 2)
            else:
                raise ValueError("Diagram must have shape (n, 2)")
        
        self._diagram = diagram
        
        # Determine t range
        if t_range is not None:
            t_min, t_max = t_range
        elif len(diagram) > 0:
            t_min = np.min(diagram[:, 0])
            t_max = np.max(diagram[:, 1])
            # Add padding
            padding = 0.1 * (t_max - t_min) if t_max > t_min else 0.1
            t_min -= padding
            t_max += padding
        else:
            t_min, t_max = 0, 1
        
        self.t_values_ = np.linspace(t_min, t_max, self.resolution)
        
        # Compute landscapes
        self.landscapes_ = self._compute_landscapes(diagram)
        
        self._is_fitted = True
        return self
    
    def _compute_landscapes(self, diagram: np.ndarray) -> np.ndarray:
        """
        Compute persistence landscape functions.
        
        The k-th landscape λₖ(t) is the k-th largest value of the
        tent functions at point t.
        """
        landscapes = np.zeros((self.n_landscapes, self.resolution))
        
        if len(diagram) == 0:
            return landscapes
        
        for i, t in enumerate(self.t_values_):
            # Compute tent function values at t
            tent_values = []
            
            for birth, death in diagram:
                if birth <= t <= death:
                    # Tent function: rises from birth to midpoint, falls to death
                    midpoint = (birth + death) / 2
                    if t <= midpoint:
                        value = t - birth
                    else:
                        value = death - t
                    tent_values.append(value)
            
            # Sort in descending order
            tent_values.sort(reverse=True)
            
            # Assign to landscapes
            for k in range(min(self.n_landscapes, len(tent_values))):
                landscapes[k, i] = tent_values[k]
        
        return landscapes
    
    def transform(self, diagram: np.ndarray = None) -> np.ndarray:
        """
        Return computed landscapes.
        
        Parameters
        ----------
        diagram : ignored if already fitted
            
        Returns
        -------
        np.ndarray of shape (n_landscapes, resolution)
            Landscape functions
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        return self.landscapes_
    
    def fit_transform(
        self, 
        diagram: np.ndarray,
        t_range: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Fit and return landscapes in one step.
        
        Parameters
        ----------
        diagram : np.ndarray
            Persistence diagram
        t_range : tuple, optional
            Filtration range
            
        Returns
        -------
        np.ndarray
            Landscape functions
        """
        return self.fit(diagram, t_range).transform()
    
    def get_landscapes(self) -> np.ndarray:
        """
        Get computed landscape functions.
        
        Returns
        -------
        np.ndarray of shape (n_landscapes, resolution)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        return self.landscapes_.copy()
    
    def get_landscape(self, k: int = 0) -> np.ndarray:
        """
        Get k-th landscape function.
        
        Parameters
        ----------
        k : int
            Landscape index (0-indexed, so k=0 is λ₁)
            
        Returns
        -------
        np.ndarray of shape (resolution,)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        if k >= self.n_landscapes:
            raise ValueError(f"k={k} >= n_landscapes={self.n_landscapes}")
        return self.landscapes_[k].copy()
    
    def get_t_values(self) -> np.ndarray:
        """Get filtration values where landscape is evaluated."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        return self.t_values_.copy()
    
    def compute_norm(self, p: float = 2, k: int = 0) -> float:
        """
        Compute Lᵖ norm of k-th landscape.
        
        Parameters
        ----------
        p : float
            Order of norm (p=2 for L² norm)
        k : int
            Landscape index
            
        Returns
        -------
        float
            Norm value
        """
        landscape = self.get_landscape(k)
        dt = self.t_values_[1] - self.t_values_[0] if len(self.t_values_) > 1 else 1
        
        if p == np.inf:
            return float(np.max(np.abs(landscape)))
        else:
            return float(np.power(np.sum(np.power(np.abs(landscape), p)) * dt, 1/p))
    
    def compute_inner_product(
        self, 
        other: 'PersistenceLandscape',
        k: int = 0
    ) -> float:
        """
        Compute inner product with another landscape.
        
        Parameters
        ----------
        other : PersistenceLandscape
            Another fitted landscape
        k : int
            Landscape index
            
        Returns
        -------
        float
            Inner product value
        """
        if not self._is_fitted or not other._is_fitted:
            raise RuntimeError("Both landscapes must be fitted")
        
        if self.resolution != other.resolution:
            raise ValueError("Landscapes must have same resolution")
        
        l1 = self.get_landscape(k)
        l2 = other.get_landscape(k)
        
        dt = self.t_values_[1] - self.t_values_[0] if len(self.t_values_) > 1 else 1
        
        return float(np.sum(l1 * l2) * dt)
    
    def compute_distance(
        self, 
        other: 'PersistenceLandscape',
        p: float = 2
    ) -> float:
        """
        Compute Lᵖ distance to another landscape.
        
        Parameters
        ----------
        other : PersistenceLandscape
            Another fitted landscape
        p : float
            Order of norm
            
        Returns
        -------
        float
            Distance value
        """
        if not self._is_fitted or not other._is_fitted:
            raise RuntimeError("Both landscapes must be fitted")
        
        total_distance = 0
        
        for k in range(min(self.n_landscapes, other.n_landscapes)):
            diff = self.get_landscape(k) - other.get_landscape(k)
            dt = self.t_values_[1] - self.t_values_[0] if len(self.t_values_) > 1 else 1
            
            if p == np.inf:
                total_distance += np.max(np.abs(diff))
            else:
                total_distance += np.power(np.sum(np.power(np.abs(diff), p)) * dt, 1/p)
        
        return float(total_distance)
    
    def extract_features(self) -> dict:
        """
        Extract summary features from landscapes.
        
        Returns
        -------
        dict
            Dictionary of landscape features
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        
        features = {}
        
        for k in range(self.n_landscapes):
            landscape = self.landscapes_[k]
            prefix = f"landscape_{k+1}"
            
            features[f"{prefix}_max"] = float(np.max(landscape))
            features[f"{prefix}_mean"] = float(np.mean(landscape))
            features[f"{prefix}_std"] = float(np.std(landscape))
            features[f"{prefix}_auc"] = float(np.trapz(landscape, self.t_values_))
            features[f"{prefix}_l2_norm"] = self.compute_norm(p=2, k=k)
        
        return features
    
    def to_vector(self) -> np.ndarray:
        """
        Flatten landscapes into a single feature vector.
        
        Returns
        -------
        np.ndarray
            Flattened landscape vector
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() first")
        return self.landscapes_.flatten()
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"PersistenceLandscape("
            f"resolution={self.resolution}, "
            f"n_landscapes={self.n_landscapes}, "
            f"status={status})"
        )


def average_landscapes(
    landscapes_list: List[PersistenceLandscape]
) -> np.ndarray:
    """
    Compute average of multiple persistence landscapes.
    
    Parameters
    ----------
    landscapes_list : list of PersistenceLandscape
        Fitted landscape objects
        
    Returns
    -------
    np.ndarray
        Average landscape
    """
    if not landscapes_list:
        raise ValueError("Empty list")
    
    arrays = [pl.get_landscapes() for pl in landscapes_list]
    return np.mean(arrays, axis=0)


def landscape_distance_matrix(
    landscapes_list: List[PersistenceLandscape],
    p: float = 2
) -> np.ndarray:
    """
    Compute pairwise distance matrix between landscapes.
    
    Parameters
    ----------
    landscapes_list : list of PersistenceLandscape
        Fitted landscape objects
    p : float
        Order of norm for distance
        
    Returns
    -------
    np.ndarray of shape (n, n)
        Distance matrix
    """
    n = len(landscapes_list)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = landscapes_list[i].compute_distance(landscapes_list[j], p=p)
            distances[i, j] = d
            distances[j, i] = d
    
    return distances
