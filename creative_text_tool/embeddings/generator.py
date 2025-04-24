from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name="AnnaWegmann/Style-Embedding"):
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self):
        """Load the embedding model"""
        # model = AutoModel.from_pretrained('AnnaWegmann/Style-Embedding')
        model = SentenceTransformer(self.model_name)
        
        return model
        
    def generate_embeddings(self, sentences):
        return self.model.encode(sentences)

    def compute_covariance_matrix(self, embeddings, normalize=True):
        # Normalize embeddings if requested
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        # Center the embeddings
        embeddings_centered = embeddings - np.mean(embeddings, axis=0)
        
        # Compute covariance matrix
        n_samples = embeddings.shape[0]
        cov_matrix = np.dot(embeddings_centered.T, embeddings_centered) / (n_samples - 1)

        return cov_matrix