import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compare_embeddings(embeddings1, embeddings2, top_n=20):
    """
    Compare two sets of embeddings row by row and analyze their differences.
    
    Parameters:
    -----------
    embeddings1 : numpy.ndarray
        First array of embeddings with shape (n_samples, n_dimensions)
    embeddings2 : numpy.ndarray
        Second array of embeddings with shape (n_samples, n_dimensions)
    top_n : int, optional
        Number of dimensions with largest differences to display
        
    Returns:
    --------
    diff_stats : dict
        Dictionary containing various statistics about the differences
    """
    # Ensure embeddings have the same shape
    if embeddings1.shape != embeddings2.shape:
        raise ValueError(f"Embedding arrays must have the same shape. Got {embeddings1.shape} and {embeddings2.shape}")
    
    n_samples, n_dimensions = embeddings1.shape
    
    # Calculate absolute differences for each dimension
    abs_diff = np.abs(embeddings1 - embeddings2)
    
    # Calculate mean absolute difference for each dimension
    dimension_diff = np.mean(abs_diff, axis=0)
    
    # Calculate overall statistics
    total_diff = np.mean(abs_diff)
    max_diff = np.max(dimension_diff)
    min_diff = np.min(dimension_diff)
    
    # Find dimensions with largest differences
    top_diff_indices = np.argsort(dimension_diff)[-top_n:][::-1]
    top_diff_values = dimension_diff[top_diff_indices]
    
    # Calculate similarity metrics for each pair of sentences
    cosine_similarities = np.zeros(n_samples)
    for i in range(n_samples):
        cosine_similarities[i] = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
    
    # Compile statistics
    diff_stats = {
        'mean_abs_diff': total_diff,
        'max_dimension_diff': max_diff,
        'min_dimension_diff': min_diff,
        'top_diff_dimensions': list(zip(top_diff_indices, top_diff_values)),
        'mean_cosine_similarity': np.mean(cosine_similarities),
        'min_cosine_similarity': np.min(cosine_similarities),
        'dimension_differences': dimension_diff,
        'sentence_differences': np.mean(abs_diff, axis=1),
        'cosine_similarities': cosine_similarities
    }
    
    return diff_stats