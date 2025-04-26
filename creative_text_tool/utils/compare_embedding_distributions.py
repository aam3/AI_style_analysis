import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

from sklearn.manifold import TSNE

def reduce_dimensions(embeddings, n_components=1, perplexity=15, random_state=42):
    """
    Reduce high-dimensional embeddings to 1D using t-SNE
    
    Parameters:
    embeddings (numpy.ndarray): Matrix of embeddings, shape (n_samples, n_dimensions)
    n_components (int): Number of dimensions in output (default: 1)
    perplexity (float): t-SNE perplexity parameter (default: 30)
    random_state (int): Random seed for reproducibility
    
    Returns:
    numpy.ndarray: Reduced embeddings, shape (n_samples, n_components)
    """
    # Adjust perplexity for 1D projection if needed
    # Perplexity should be smaller than the number of points and typically between 5-50
    if n_components==1:
        perplexity = min(perplexity, embeddings.shape[0] // 3)
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                random_state=random_state, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)
    return reduced_embeddings

def compare_embedding_distributions(embeddings_a, embeddings_b, labels, grid_size=100, plot=True):
    """
    Compare two sets of embeddings by reducing dimensions and calculating distribution divergence
    
    Parameters:
    embeddings_a (numpy.ndarray): First set of embeddings, shape (n_samples_a, n_dimensions)
    embeddings_b (numpy.ndarray): Second set of embeddings, shape (n_samples_b, n_dimensions)
    grid_size (int): Size of the grid for density estimation (default: 100)
    plot (bool): Whether to generate visualization plots (default: True)
    
    Returns:
    dict: Dictionary containing:
        - 'kl_divergence': KL divergence from B to A
        - 'symmetric_kl': Symmetric KL divergence
        - 'js_divergence': Jensen-Shannon divergence
        - 'reduced_a': Reduced embeddings for A
        - 'reduced_b': Reduced embeddings for B
        - 'density_a': Density estimate for A
        - 'density_b': Density estimate for B
    """
    # Reduce dimensions
    reduced_a = reduce_dimensions(embeddings_a, n_components=2)
    reduced_b = reduce_dimensions(embeddings_b, n_components=2)
    
    # Estimate densities
    xx_a, yy_a, density_a = estimate_density(reduced_a, grid_size=grid_size)
    xx_b, yy_b, density_b = estimate_density(reduced_b, grid_size=grid_size)
    
    # Ensure both densities are evaluated on the same grid
    # We'll use the grid from A for both
    if not np.array_equal(xx_a, xx_b) or not np.array_equal(yy_a, yy_b):
        # Recompute density_b on the same grid as density_a
        x_min, x_max = reduced_a[:, 0].min(), reduced_a[:, 0].max()
        y_min, y_max = reduced_a[:, 1].min(), reduced_a[:, 1].max()
        
        # Add a margin
        margin_x = (x_max - x_min) * 0.1
        margin_y = (y_max - y_min) * 0.1
        x_min -= margin_x
        x_max += margin_x
        y_min -= margin_y
        y_max += margin_y
        
        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        
        kde_b = stats.gaussian_kde(reduced_b.T)
        density_b = kde_b(grid_points.T).reshape(grid_size, grid_size)
        density_b = density_b / density_b.sum()
        
        xx_b, yy_b = xx, yy
    
    # Calculate divergences
    kl_div_b_to_a = kl_divergence_2d(density_a, density_b)
    kl_div_a_to_b = kl_divergence_2d(density_b, density_a)
    sym_kl = 0.5 * (kl_div_a_to_b + kl_div_b_to_a)
    js_div = js_divergence_2d(density_a, density_b)
    
    # Visualization if requested
    if plot:
        plt.figure(figsize=(15, 6))
        
        # Plot reduced embeddings
        plt.subplot(1, 2, 1)
        plt.scatter(reduced_a[:, 0], reduced_a[:, 1], alpha=0.5, label=labels[0])
        plt.scatter(reduced_b[:, 0], reduced_b[:, 1], alpha=0.5, label=labels[1])
        plt.title('Reduced Embeddings')
        plt.legend()
        
        # Combined density plot with alpha blending to show overlap
        plt.subplot(1, 2, 2)
        
        # Find common min/max for density scales
        vmin = min(density_a.min(), density_b.min())
        vmax = max(density_a.max(), density_b.max())
        
        # Create same-scale levels for consistent contours
        levels = np.linspace(vmin, vmax, 11)
        
        # Use alpha blending for better visualization of overlap
        contour_a = plt.contourf(xx_a, yy_a, density_a, cmap='Blues', alpha=0.5, levels=levels)
        contour_b = plt.contourf(xx_b, yy_b, density_b, cmap='Reds', alpha=0.5, levels=levels)
        
        # Add contour lines for better edge visibility
        plt.contour(xx_a, yy_a, density_a, colors='blue', levels=levels[::2], linewidths=0.5)
        plt.contour(xx_b, yy_b, density_b, colors='red', levels=levels[::2], linewidths=0.5)
        
        # Create a single shared colorbar for both distributions
        cbar = plt.colorbar(contour_a, label='Density', shrink=0.7, pad=0.01)
        
        # Add legend to identify distributions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.5, label=labels[0]),
            Patch(facecolor='red', alpha=0.5, label=labels[1])
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.title(f'Combined Density Distributions\nKL(B||A): {kl_div_b_to_a:.4f}, KL(A||B): {kl_div_a_to_b:.4f}, JS: {js_div:.4f}')
        
        plt.tight_layout()
        plt.show()
    
    return {
        'kl_divergence_b_to_a': kl_div_b_to_a,
        'kl_divergence_a_to_b': kl_div_a_to_b,
        'symmetric_kl': sym_kl,
        'js_divergence': js_div,
        'reduced_a': reduced_a,
        'reduced_b': reduced_b,
        'density_a': density_a,
        'density_b': density_b
    }

def estimate_density(points, grid_size=100, bandwidth=None):
    """
    Estimate 2D probability density using Gaussian KDE
    
    Parameters:
    points (numpy.ndarray): 2D points, shape (n_samples, 2)
    grid_size (int): Size of the grid for density estimation (default: 100)
    bandwidth (float or None): KDE bandwidth parameter (default: None, auto-estimation)
    
    Returns:
    tuple: (x_grid, y_grid, density)
        x_grid, y_grid: Meshgrid of evaluation points
        density: 2D array of density values
    """
    # Define the grid for evaluation
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    
    # Add a margin to the grid boundaries
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    # Perform KDE
    kde = stats.gaussian_kde(points.T, bw_method=bandwidth)
    density = kde(grid_points.T).reshape(grid_size, grid_size)
    
    # Normalize to ensure it integrates to 1
    density = density / density.sum()
    
    return xx, yy, density    

def kl_divergence_2d(p, q):
    """
    Calculate Kullback-Leibler divergence between two probability distributions
    
    Parameters:
    p (numpy.ndarray): First probability distribution (2D array)
    q (numpy.ndarray): Second probability distribution (2D array)
    
    Returns:
    float: KL divergence from q to p
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize to ensure they sum to 1
    p = p / p.sum()
    q = q / q.sum()
    
    # Calculate KL divergence: sum(p * log(p/q))
    kl_div = np.sum(p * np.log(p / q))
    
    return kl_div  

def js_divergence_2d(p, q):
    """
    Calculate Jensen-Shannon divergence between two probability distributions
    
    Parameters:
    p (numpy.ndarray): First probability distribution (2D array)
    q (numpy.ndarray): Second probability distribution (2D array)
    
    Returns:
    float: JS divergence between p and q
    """
    # Flatten and normalize
    p_flat = p.flatten()
    q_flat = q.flatten()
    p_flat = p_flat / p_flat.sum()
    q_flat = q_flat / q_flat.sum()
    
    return jensenshannon(p_flat, q_flat, base=2) ** 2
