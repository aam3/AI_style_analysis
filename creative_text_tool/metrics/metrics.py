import os
import numpy as np
from typing import Tuple, Dict, List, Union, Optional
from sklearn.metrics.pairwise import cosine_similarity

class Metrics:
    # def __init__(self):
    #     self._compute_center_embedding_by_window
        
    # def _init_client(self):
    #     """Initialize the Claude API client"""
    #     return anthropic.Anthropic(api_key=self.api_key)

    def compute_mean_similarity_by_window(self, embeddings: np.ndarray, window_size: int = 1, normalize: bool = True, centroid = False ) -> Tuple[float, Dict]:
        
        if window_size is not None and window_size < 1:
            raise ValueError("Window size must be at least 1 or None for all pairs")
        
        # Normalize embeddings if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        # Calculate similarities within windows
        n_sentences = len(embeddings)
        similarities = []
        
        # If window_size is None, compute all pairwise similarities
        if window_size is None:
            # Get full similarity matrix
            sim_matrix = cosine_similarity(embeddings)
            # Extract upper triangle to avoid duplicates and self-similarities
            indices = np.triu_indices(n_sentences, k=1)
            similarities = sim_matrix[indices]
            return similarities

        if centroid:
            centroids = self._compute_center_embedding_by_window(embeddings, window_size=window_size, normalize=normalize )

            for i in range(n_sentences):
                # Define window boundaries
                start = max(0, i - window_size)
                end = min(n_sentences, i + window_size + 1)
                
                # Skip self-comparison (i == j)
                window_indices = [j for j in range(start, end)]
                
                if window_indices:
                    # Calculate similarities for all sentences in window
                    current_similarities = np.mean( cosine_similarity( centroids[i].reshape(1, -1), embeddings[window_indices] ).flatten() )
                    
                    similarities.append(current_similarities)            
        else:
            for i in range(n_sentences):
                # Define window boundaries
                start = max(0, i - window_size)
                end = min(n_sentences, i + window_size + 1)
                
                # Skip self-comparison (i == j)
                window_indices = [j for j in range(start, end) if j != i]
                
                if window_indices:
                    # Calculate similarities for all sentences in window
                    current_similarities = np.mean(cosine_similarity(embeddings[i].reshape(1, -1), embeddings[window_indices]).flatten())
                    
                    similarities.append(current_similarities)
    
        return similarities

    def _compute_center_embedding_by_window(self, embeddings: np.ndarray, window_size: int = 1, normalize: bool = True ) -> Tuple[float, Dict]:
        
        if window_size is None or window_size < 1:
            raise ValueError("Window size must be at least 1.")
        
        # Normalize embeddings if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        # Calculate similarities within windows
        n_sentences = len(embeddings)
        centroids = []
        
        for i in range(n_sentences):
            # Define window boundaries
            start = max(0, i - window_size)
            end = min(n_sentences, i + window_size + 1)
            
            # Skip self-comparison (i == j)
            window_indices = [j for j in range(start, end)]
            
            if window_indices:
                centroid = np.mean(embeddings[window_indices], axis=0).reshape([1,-1])
                centroids.append(centroid)                

        centroids = np.concatenate(centroids, axis=0)        
    
        return centroids


Metrics = Metrics()