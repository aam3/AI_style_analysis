from .document import Document
from .text_generator import TextGenerator
from .text_processor import TextProcessor
from .embeddings.generator import EmbeddingGenerator
# from .metrics.compute_covariance_matrix import compute_covariance_matrix
# from .compute_sentence_similarities_by_window import compute_sentence_similarities_by_window

class CreativeTextTool:
    def __init__(self, api_key=None, model_name="claude-3-7-sonnet-20250219"):
        self.text_generator = TextGenerator(api_key, model_name)
        self.text_processor = TextProcessor()
        self.embedding_generator = EmbeddingGenerator()
        # self.dimension_reducer = DimensionReducer()
        # self.metrics_calculator = MetricsCalculator()

    def return_text_from_prompt(self, messages, system_prompt=""):
        """Generate text and create a Document from it."""
        # Generate text
        raw_text = self.text_generator.generate(messages, system_prompt)
        
        # Create and process document
        return raw_text     
        
    def create_document_from_prompt(self, messages, doc_name, system_prompt=""):
        """Generate text and create a Document from it."""
        # Generate text
        raw_text = self.text_generator.generate(messages, system_prompt)
        
        # Create and process document
        return self.create_document(raw_text)

    def create_document(self, text, doc_name):
        """Create a Document from existing text."""
        # Initialize document
        document = Document(text, doc_name)
        
        # Process the document
        self._process_document(document)
        
        return document        
        
    def _process_document(self, document):
        """Process already generated text"""
        # Break into sentences
        document.sentences = self.text_processor.tokenize_sentences(document.text)
        
        # Generate embeddings
        document.embeddings = self.embedding_generator.generate_embeddings(document.sentences)
        
        # # Reduce dimensions
        # document.reduced_embeddings = self.dimension_reducer.reduce(document.embeddings)
        
        # # Calculate metrics
        # document.metrics = self.metrics_calculator.calculate(
        #     document.sentences, 
        #     document.embeddings, 
        #     document.reduced_embeddings
        # )