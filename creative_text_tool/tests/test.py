import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

from .compare_embeddings import compare_embeddings
from .visualize_embedding_differences import visualize_embedding_differences

class Test:
    # def __init__(self):
    #     self.text_generator = TextGenerator(api_key, model_name)
    #     self.text_processor = TextProcessor()
    #     self.embedding_generator = EmbeddingGenerator()

    @staticmethod
    def _calculate_embeddings_differences(embeddings1, embeddings2, top_n):
        return(compare_embeddings(embeddings1, embeddings2, top_n=top_n))

    @staticmethod
    def _visualize_embedding_differences(diff_stats, title="Embedding Differences"):
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
        top_dims = sorted(top_dims, key=lambda tup: tup[1], reverse=True)
        dims = [d[0].astype(str) for d in top_dims]
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

    def compare_embedding_sets(self, original_embeddings, modified_embeddings, top_n=20, plot=True):
           
        print("\nComparing style embeddings:")
        original_diff_stats = self._calculate_embeddings_differences(original_embeddings, original_embeddings, top_n)
        style_diff_stats = self._calculate_embeddings_differences(original_embeddings, modified_embeddings, top_n)
        
        print(f"Mean absolute difference: {style_diff_stats['mean_abs_diff']:.4f}")
        print(f"Mean cosine similarity: {style_diff_stats['mean_cosine_similarity']:.4f}")
        
        # Calculate improvement in similarity
        similarity_improvement = style_diff_stats['mean_cosine_similarity'] - original_diff_stats['mean_cosine_similarity']
        print(f"\nImprovement in cosine similarity: {similarity_improvement:.4f}")
        
        # Visualize differences
        if plot:
            self._visualize_embedding_differences(style_diff_stats, "Style Embedding Differences")

        return style_diff_stats

    @staticmethod
    def apply_embeddings_mask(embeddings, dimensions_to_mask):
    
        mask = np.ones_like(embeddings)
        # Handle both single embedding and batched embeddings
        if mask.ndim == 1:
            mask[dimensions_to_mask] = 0
        else:
            mask[:, dimensions_to_mask] = 0
            
        # Apply mask by element-wise multiplication
        embeddings = embeddings * mask
    
        return(embeddings)            
            
Test = Test()