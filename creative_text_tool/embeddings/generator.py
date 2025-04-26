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