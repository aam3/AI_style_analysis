import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def visualize_embedding_differences(diff_stats, title="Embedding Differences"):
    """
    Visualize the differences between two sets of embeddings.
    
    Parameters:
    -----------
    diff_stats : dict
        Dictionary containing difference statistics from compare_embeddings()
    title : str, optional
        Title for the plots
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    
    # Plot 1: Distribution of dimension differences
    sns.histplot(diff_stats['dimension_differences'], bins=50, kde=True, ax=axes[0])
    axes[0].set_title('Distribution of Mean Absolute Differences Across Dimensions')
    axes[0].set_xlabel('Mean Absolute Difference')
    axes[0].set_ylabel('Count')
    
    # Plot 2: Top dimensions with differences
    top_dims = diff_stats['top_diff_dimensions']
    dims = [d[0] for d in top_dims]
    values = [d[1] for d in top_dims]
    
    sns.barplot(x=dims, y=values, ax=axes[1])
    axes[1].set_title(f'Top {len(top_dims)} Dimensions with Largest Differences')
    axes[1].set_xlabel('Dimension Index')
    axes[1].set_ylabel('Mean Absolute Difference')
    axes[1].tick_params(axis='x', rotation=45)
        
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.96, 1])
    plt.savefig("embedding_differences.png")
    plt.show()