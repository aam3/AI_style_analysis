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

def compare_embedding_distributions_1d(embeddings_a, embeddings_b, labels, embeddings_c=None, grid_size=1000, plot=True):
    """
    Compare two sets of embeddings by reducing dimensions to 1D and calculating distribution divergence
    
    Parameters:
    embeddings_a (numpy.ndarray): First set of embeddings, shape (n_samples_a, n_dimensions)
    embeddings_b (numpy.ndarray): Second set of embeddings, shape (n_samples_b, n_dimensions)
    grid_size (int): Size of the grid for density estimation (default: 1000)
    plot (bool): Whether to generate visualization plots (default: True)
    
    Returns:
    dict: Dictionary containing:
        - 'kl_divergence_b_to_a': KL divergence from B to A
        - 'kl_divergence_a_to_b': KL divergence from A to B
        - 'symmetric_kl': Symmetric KL divergence
        - 'js_divergence': Jensen-Shannon divergence
        - 'reduced_a': Reduced embeddings for A
        - 'reduced_b': Reduced embeddings for B
        - 'density_a': Density estimate for A
        - 'density_b': Density estimate for B
    """
    # Reduce dimensions to 1D
    reduced_a = reduce_dimensions(embeddings_a, n_components=1)
    reduced_b = reduce_dimensions(embeddings_b, n_components=1)
    if embeddings_c is not None:
        reduced_c = reduce_dimensions(embeddings_c, n_components=1)
    
    # Estimate densities using a common grid for fair comparison
    if embeddings_c is not None:
        min_val = min(reduced_a.min(), reduced_b.min(), reduced_c.min())
        max_val = max(reduced_a.max(), reduced_b.max(), reduced_c.max())
    else:
        min_val = min(reduced_a.min(), reduced_b.min())
        max_val = max(reduced_a.max(), reduced_b.max())
        
    margin = (max_val - min_val) * 0.1
    min_val -= margin
    max_val += margin
    
    common_grid = np.linspace(min_val, max_val, grid_size)
    
    # Create KDE for both distributions
    kde_a = stats.gaussian_kde(reduced_a.flatten())
    kde_b = stats.gaussian_kde(reduced_b.flatten())
    if embeddings_c is not None:
        kde_c = stats.gaussian_kde(reduced_c.flatten())
    
    # Evaluate on common grid
    density_a = kde_a(common_grid)
    density_b = kde_b(common_grid)
    if embeddings_c is not None:
        density_c = kde_c(common_grid)
    
    # Normalize
    density_a = density_a / (density_a.sum() * (common_grid[1] - common_grid[0]))
    density_b = density_b / (density_b.sum() * (common_grid[1] - common_grid[0]))
    if embeddings_c is not None:
        density_c = density_c / (density_c.sum() * (common_grid[1] - common_grid[0]))
    
    # Calculate divergences
    kl_div_b_to_a = kl_divergence_1d(density_a, density_b)
    kl_div_a_to_b = kl_divergence_1d(density_b, density_a)
    sym_kl = 0.5 * (kl_div_a_to_b + kl_div_b_to_a)
    js_div = js_divergence_1d(density_a, density_b)
    
    # Visualization if requested
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Subplot 1: Histogram of reduced embeddings
        plt.subplot(2, 1, 1)
        plt.hist(reduced_a, bins=30, alpha=0.5, label=labels[0], density=True)
        plt.hist(reduced_b, bins=30, alpha=0.5, label=labels[1], density=True)
        if embeddings_c is not None:
            plt.hist(reduced_c, bins=30, alpha=0.5, label=labels[2], density=True)
        plt.title('Histogram of 1D Reduced Embeddings')
        plt.legend()
        
        # Subplot 2: KDE comparison
        plt.subplot(2, 1, 2)
        plt.plot(common_grid, density_a, 'b-', label=labels[0])
        plt.plot(common_grid, density_b, 'r-', label=labels[1])
        if embeddings_c is not None:
            plt.plot(common_grid, density_c, 'g-', label=labels[2])
        plt.fill_between(common_grid, density_a, density_b, 
                         where=(density_a > density_b), alpha=0.3, color='blue')
        plt.fill_between(common_grid, density_a, density_b, 
                         where=(density_b > density_a), alpha=0.3, color='red')
        if embeddings_c is not None:
            plt.fill_between(common_grid, density_a, density_c, 
                             where=(density_c > density_a), alpha=0.3, color='green')
        plt.title(f'Density Comparison\nKL(B||A): {kl_div_b_to_a:.4f}, KL(A||B): {kl_div_a_to_b:.4f}, JS: {js_div:.4f}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'kl_divergence_b_to_a': kl_div_b_to_a,
        'kl_divergence_a_to_b': kl_div_a_to_b,
        'symmetric_kl': sym_kl,
        'js_divergence': js_div,
        'reduced_a': reduced_a,
        'reduced_b': reduced_b,
        'common_grid': common_grid,
        'density_a': density_a,
        'density_b': density_b
    }


def estimate_density_1d(points, grid_size=1000, bandwidth=None):
    """
    Estimate 1D probability density using Gaussian KDE
    
    Parameters:
    points (numpy.ndarray): 1D points, shape (n_samples, 1)
    grid_size (int): Size of the grid for density estimation (default: 1000)
    bandwidth (float or None): KDE bandwidth parameter (default: None, auto-estimation)
    
    Returns:
    tuple: (x_grid, density)
        x_grid: Array of evaluation points
        density: Array of density values
    """
    # Flatten the points array
    points_flat = points.flatten()
    
    # Define the grid for evaluation
    x_min, x_max = points_flat.min(), points_flat.max()
    
    # Add a margin to the grid boundaries
    margin = (x_max - x_min) * 0.1
    x_min -= margin
    x_max += margin
    
    x_grid = np.linspace(x_min, x_max, grid_size)
    
    # Perform KDE
    kde = stats.gaussian_kde(points_flat, bw_method=bandwidth)
    density = kde(x_grid)
    
    # Normalize to ensure it integrates to 1
    density = density / (density.sum() * (x_grid[1] - x_grid[0]))
    
    return x_grid, density

def kl_divergence_1d(p, q):
    """
    Calculate Kullback-Leibler divergence between two probability distributions
    
    Parameters:
    p (numpy.ndarray): First probability distribution (1D array)
    q (numpy.ndarray): Second probability distribution (1D array)
    
    Returns:
    float: KL divergence from q to p
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # Normalize to ensure they sum to 1
    p = p / (p.sum() * (1/len(p)))  # Accounting for bin width
    q = q / (q.sum() * (1/len(q)))
    
    # Calculate KL divergence: sum(p * log(p/q))
    # Need to ensure p and q have the same length
    assert len(p) == len(q), "Distributions must have same number of points"
    kl_div = np.sum(p * np.log(p / q)) * (1/len(p))  # Accounting for bin width
    
    return kl_div

def js_divergence_1d(p, q):
    """
    Calculate Jensen-Shannon divergence between two probability distributions
    
    Parameters:
    p (numpy.ndarray): First probability distribution (1D array)
    q (numpy.ndarray): Second probability distribution (1D array)
    
    Returns:
    float: JS divergence between p and q
    """
    # Normalize
    p = p / p.sum()
    q = q / q.sum()
    
    return jensenshannon(p, q, base=2) ** 2

