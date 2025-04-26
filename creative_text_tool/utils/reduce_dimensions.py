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