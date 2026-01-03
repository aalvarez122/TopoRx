"""
Visualization Module
====================

Interactive visualizations for topological biomarker analysis
using Plotly.

Author: Angelica Alvarez
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def _check_plotly():
    """Check if Plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for visualization. "
            "Install with: pip install plotly"
        )


def plot_persistence_diagram(
    diagrams: List[np.ndarray],
    dimensions: Optional[List[int]] = None,
    title: str = "Persistence Diagram",
    show_diagonal: bool = True,
    size: Tuple[int, int] = (700, 600)
) -> 'go.Figure':
    """
    Create interactive persistence diagram plot.
    
    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams for each dimension
    dimensions : list of int, optional
        Which dimensions to plot (default: all)
    title : str
        Plot title
    show_diagonal : bool
        Whether to show the diagonal line
    size : tuple
        Figure size (width, height)
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure
        
    Examples
    --------
    >>> fig = plot_persistence_diagram(diagrams)
    >>> fig.show()
    """
    _check_plotly()
    
    if dimensions is None:
        dimensions = list(range(len(diagrams)))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    dim_names = ['H₀ (Components)', 'H₁ (Loops)', 'H₂ (Voids)', 'H₃', 'H₄']
    
    fig = go.Figure()
    
    # Find global min/max for axes
    all_points = []
    for dim in dimensions:
        if dim < len(diagrams) and len(diagrams[dim]) > 0:
            all_points.extend(diagrams[dim].flatten().tolist())
    
    if all_points:
        axis_min = min(all_points) - 0.1
        axis_max = max(all_points) + 0.1
    else:
        axis_min, axis_max = 0, 1
    
    # Add diagonal line
    if show_diagonal:
        fig.add_trace(go.Scatter(
            x=[axis_min, axis_max],
            y=[axis_min, axis_max],
            mode='lines',
            line=dict(color='gray', dash='dash', width=1),
            name='Diagonal',
            showlegend=False
        ))
    
    # Add points for each dimension
    for dim in dimensions:
        if dim >= len(diagrams):
            continue
            
        diagram = diagrams[dim]
        if len(diagram) == 0:
            continue
        
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistence = deaths - births
        
        # Hover text
        hover_text = [
            f"Birth: {b:.3f}<br>Death: {d:.3f}<br>Persistence: {p:.3f}"
            for b, d, p in zip(births, deaths, persistence)
        ]
        
        fig.add_trace(go.Scatter(
            x=births,
            y=deaths,
            mode='markers',
            marker=dict(
                size=10,
                color=colors[dim % len(colors)],
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            name=dim_names[dim] if dim < len(dim_names) else f'H{dim}',
            hovertemplate="%{text}<extra></extra>",
            text=hover_text
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Birth",
        yaxis_title="Death",
        width=size[0],
        height=size[1],
        template="plotly_white",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="closest"
    )
    
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def plot_betti_curves(
    diagrams: List[np.ndarray],
    n_points: int = 100,
    title: str = "Betti Curves",
    size: Tuple[int, int] = (800, 500)
) -> 'go.Figure':
    """
    Plot Betti curves showing topological features over filtration.
    
    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams
    n_points : int
        Number of filtration values
    title : str
        Plot title
    size : tuple
        Figure size
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    dim_names = ['β₀ (Components)', 'β₁ (Loops)', 'β₂ (Voids)']
    
    # Determine filtration range
    all_values = []
    for diagram in diagrams:
        if len(diagram) > 0:
            all_values.extend(diagram.flatten().tolist())
    
    if not all_values:
        all_values = [0, 1]
    
    filt_min, filt_max = min(all_values), max(all_values)
    filtration = np.linspace(filt_min, filt_max, n_points)
    
    fig = go.Figure()
    
    for dim, diagram in enumerate(diagrams):
        if dim >= 3:
            break
            
        betti = np.zeros(n_points)
        
        if len(diagram) > 0:
            for i, f in enumerate(filtration):
                betti[i] = np.sum((diagram[:, 0] <= f) & (diagram[:, 1] > f))
        
        fig.add_trace(go.Scatter(
            x=filtration,
            y=betti,
            mode='lines',
            name=dim_names[dim],
            line=dict(color=colors[dim], width=3),
            fill='tozeroy',
            fillcolor=f'rgba{tuple(list(int(colors[dim][i:i+2], 16) for i in (1, 3, 5)) + [0.2])}'
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Filtration Value",
        yaxis_title="Betti Number",
        width=size[0],
        height=size[1],
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig


def plot_persistence_landscape(
    landscapes: np.ndarray,
    t_values: np.ndarray,
    n_landscapes: int = 3,
    title: str = "Persistence Landscapes",
    size: Tuple[int, int] = (800, 500)
) -> 'go.Figure':
    """
    Plot persistence landscape functions.
    
    Parameters
    ----------
    landscapes : np.ndarray
        Landscape array of shape (n_landscapes, resolution)
    t_values : np.ndarray
        Filtration values
    n_landscapes : int
        Number of landscapes to plot
    title : str
        Plot title
    size : tuple
        Figure size
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    colors = px.colors.sequential.Viridis
    
    fig = go.Figure()
    
    for k in range(min(n_landscapes, len(landscapes))):
        color_idx = int(k * (len(colors) - 1) / max(n_landscapes - 1, 1))
        
        fig.add_trace(go.Scatter(
            x=t_values,
            y=landscapes[k],
            mode='lines',
            name=f'λ{k+1}',
            line=dict(color=colors[color_idx], width=2),
            fill='tozeroy' if k == 0 else None
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Filtration Value (t)",
        yaxis_title="Landscape Value λ(t)",
        width=size[0],
        height=size[1],
        template="plotly_white",
        hovermode="x unified"
    )
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 15,
    title: str = "Top Predictive Topological Features",
    size: Tuple[int, int] = (800, 600)
) -> 'go.Figure':
    """
    Plot feature importance as horizontal bar chart.
    
    Parameters
    ----------
    feature_names : list of str
        Names of features
    importance_scores : np.ndarray
        Importance scores
    top_n : int
        Number of top features to show
    title : str
        Plot title
    size : tuple
        Figure size
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    # Sort and get top features
    indices = np.argsort(importance_scores)[::-1][:top_n]
    top_names = [feature_names[i] for i in indices]
    top_scores = importance_scores[indices]
    
    # Color by feature type
    colors = []
    for name in top_names:
        if 'entropy' in name.lower():
            colors.append('#9b59b6')  # Purple for entropy
        elif 'landscape' in name.lower():
            colors.append('#3498db')  # Blue for landscape
        elif 'betti' in name.lower():
            colors.append('#2ecc71')  # Green for Betti
        elif 'H0' in name:
            colors.append('#e74c3c')  # Red for H0
        elif 'H1' in name:
            colors.append('#f39c12')  # Orange for H1
        else:
            colors.append('#1abc9c')  # Teal for others
    
    fig = go.Figure(go.Bar(
        x=top_scores[::-1],
        y=top_names[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(width=1, color='white')
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="Importance Score",
        yaxis_title="",
        width=size[0],
        height=size[1],
        template="plotly_white"
    )
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: Optional[float] = None,
    title: str = "ROC Curve",
    size: Tuple[int, int] = (600, 600)
) -> 'go.Figure':
    """
    Plot ROC curve.
    
    Parameters
    ----------
    fpr : np.ndarray
        False positive rates
    tpr : np.ndarray
        True positive rates
    auc_score : float, optional
        AUC score to display
    title : str
        Plot title
    size : tuple
        Figure size
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    label = f"TopoRx (AUC = {auc_score:.3f})" if auc_score else "TopoRx"
    
    fig = go.Figure()
    
    # Diagonal reference
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='Random',
        showlegend=True
    ))
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        line=dict(color='#3498db', width=3),
        name=label,
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        width=size[0],
        height=size[1],
        template="plotly_white",
        legend=dict(x=0.6, y=0.1)
    )
    
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig


def plot_comparison_results(
    results: Dict[str, Dict],
    title: str = "TDA vs Traditional Features Comparison",
    size: Tuple[int, int] = (800, 500)
) -> 'go.Figure':
    """
    Plot comparison between TDA and gene-based approaches.
    
    Parameters
    ----------
    results : dict
        Results from ComparativeAnalysis.compare()
    title : str
        Plot title
    size : tuple
        Figure size
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    methods = ['Gene-Based', 'Topological (TDA)', 'Combined']
    keys = ['gene_based', 'topological', 'combined']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    means = []
    stds = []
    
    for key in keys:
        if key in results:
            means.append(results[key]['mean_score'])
            stds.append(results[key]['std_score'])
        else:
            means.append(0)
            stds.append(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=methods,
        y=means,
        error_y=dict(type='data', array=stds, visible=True),
        marker=dict(
            color=colors,
            line=dict(width=2, color='white')
        ),
        text=[f'{m:.3f}' for m in means],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>ROC-AUC: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>"
    ))
    
    # Add improvement annotation
    if 'improvement' in results:
        imp = results['improvement']['relative_percent']
        fig.add_annotation(
            x=1,
            y=means[1] + stds[1] + 0.05,
            text=f"+{imp:.1f}% vs genes",
            showarrow=False,
            font=dict(size=14, color='#27ae60')
        )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        yaxis_title="ROC-AUC Score",
        width=size[0],
        height=size[1],
        template="plotly_white",
        showlegend=False,
        yaxis=dict(range=[0, 1.1])
    )
    
    return fig


def create_dashboard(
    diagrams: List[np.ndarray],
    feature_names: List[str],
    feature_importance: np.ndarray,
    comparison_results: Dict,
    roc_data: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
) -> 'go.Figure':
    """
    Create comprehensive dashboard with all visualizations.
    
    Parameters
    ----------
    diagrams : list of np.ndarray
        Persistence diagrams
    feature_names : list of str
        Feature names
    feature_importance : np.ndarray
        Feature importance scores
    comparison_results : dict
        Results from comparative analysis
    roc_data : tuple, optional
        (fpr, tpr, auc) for ROC curve
        
    Returns
    -------
    plotly.graph_objects.Figure
    """
    _check_plotly()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Persistence Diagram',
            'Model Comparison',
            'Betti Curves',
            'Feature Importance'
        ),
        specs=[
            [{"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    dim_names = ['H₀', 'H₁', 'H₂']
    
    # 1. Persistence Diagram (top-left)
    all_points = []
    for diagram in diagrams:
        if len(diagram) > 0:
            all_points.extend(diagram.flatten().tolist())
    
    if all_points:
        axis_range = [min(all_points) - 0.1, max(all_points) + 0.1]
    else:
        axis_range = [0, 1]
    
    # Diagonal
    fig.add_trace(
        go.Scatter(x=axis_range, y=axis_range, mode='lines',
                   line=dict(color='gray', dash='dash'), showlegend=False),
        row=1, col=1
    )
    
    for dim, diagram in enumerate(diagrams[:3]):
        if len(diagram) > 0:
            fig.add_trace(
                go.Scatter(x=diagram[:, 0], y=diagram[:, 1], mode='markers',
                          marker=dict(size=8, color=colors[dim]),
                          name=dim_names[dim]),
                row=1, col=1
            )
    
    # 2. Model Comparison (top-right)
    methods = ['Genes', 'TDA', 'Combined']
    keys = ['gene_based', 'topological', 'combined']
    bar_colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    means = [comparison_results.get(k, {}).get('mean_score', 0) for k in keys]
    
    fig.add_trace(
        go.Bar(x=methods, y=means, marker_color=bar_colors,
               text=[f'{m:.2f}' for m in means], textposition='outside',
               showlegend=False),
        row=1, col=2
    )
    
    # 3. Betti Curves (bottom-left)
    if all_points:
        filt = np.linspace(min(all_points), max(all_points), 50)
        for dim, diagram in enumerate(diagrams[:3]):
            betti = np.zeros(50)
            if len(diagram) > 0:
                for i, f in enumerate(filt):
                    betti[i] = np.sum((diagram[:, 0] <= f) & (diagram[:, 1] > f))
            fig.add_trace(
                go.Scatter(x=filt, y=betti, mode='lines',
                          line=dict(color=colors[dim], width=2),
                          name=f'β{dim}', showlegend=False),
                row=2, col=1
            )
    
    # 4. Feature Importance (bottom-right)
    top_n = 8
    indices = np.argsort(feature_importance)[::-1][:top_n]
    top_names = [feature_names[i][:15] for i in indices]
    top_scores = feature_importance[indices]
    
    fig.add_trace(
        go.Bar(x=top_scores[::-1], y=top_names[::-1], orientation='h',
               marker_color='#3498db', showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1100,
        title=dict(text="TopoRx: Topological Biomarker Discovery Dashboard",
                   font=dict(size=22)),
        template="plotly_white",
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


# Matplotlib fallback for environments without Plotly
def plot_persistence_diagram_mpl(
    diagrams: List[np.ndarray],
    ax=None,
    title: str = "Persistence Diagram"
):
    """
    Matplotlib version of persistence diagram plot.
    
    Use this if Plotly is not available.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    labels = ['H₀', 'H₁', 'H₂']
    
    all_points = []
    for diagram in diagrams:
        if len(diagram) > 0:
            all_points.extend(diagram.flatten().tolist())
    
    if all_points:
        lim = [min(all_points) - 0.1, max(all_points) + 0.1]
    else:
        lim = [0, 1]
    
    ax.plot(lim, lim, 'k--', alpha=0.3, label='Diagonal')
    
    for dim, diagram in enumerate(diagrams[:3]):
        if len(diagram) > 0:
            ax.scatter(diagram[:, 0], diagram[:, 1],
                      c=colors[dim], label=labels[dim],
                      s=50, alpha=0.7, edgecolors='white')
    
    ax.set_xlabel('Birth')
    ax.set_ylabel('Death')
    ax.set_title(title)
    ax.legend()
    ax.set_aspect('equal')
    
    return ax
