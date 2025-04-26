import numpy as np

class Document:
    """Represents a processed text document with accessible attributes."""
    
    def __init__(self, raw_text, doc_name, title=None):
        self.text = raw_text
        self.name = doc_name
        self.sentences = []
        self.embeddings = None
        self.reduced_embeddings = None
        self.metrics = {}
        self._sentence_index = {}  # For quick access to sentence position
        
    def __str__(self):
        """String representation of the document."""
        return f"{self.title}: {len(self.sentences)} sentences"
        
    def __repr__(self):
        """Formal representation of the document."""
        return f"Document(title='{self.title}', sentences={len(self.sentences)}, has_embeddings={self.embeddings is not None})"
    
    def save(self, filepath: str) -> None:
        """Save the document to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Document':
        """Load a document from a file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
        
       