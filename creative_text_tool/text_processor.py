import os
import spacy

class TextProcessor:
    def __init__(self, language="english"):
        self.language = language
        self.nlp = self._init_nlp()
        
    def _init_nlp(self):
        """Initialize NLP tools for text processing"""
        try:
            # Load English language model (run once)
            nlp = spacy.load('en_core_web_sm')
            nlp.max_length = 5000000
        except:
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load('en_core_web_sm')
            nlp.max_length = 5000000
        return nlp   
    
    def tokenize_sentences(self, text):
        """Break text into sentences"""

        doc=self.nlp(text)
        
        sentences = []
        for sent in doc.sents:
            sentences.append(sent)
    
        return(sentences)