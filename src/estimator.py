import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import RegexpStemmer

class CustomVectorizer(TfidfVectorizer):
    """TfidVectorizer."""
    
    def build_analyzer(self):
        """Build analyzer with preprocessing step - stemming"""
        
        stop_words = self.get_stop_words()

        def analyzer(doc):
            
            doc_clean = doc.lower()
            tokens = re.findall(self.token_pattern, doc_clean)
            
            regexp = RegexpStemmer('ing$|s$|ed$|able$', min=4)
            stemmed_tokens = [regexp.stem(token) for token in tokens]
            # stemmed_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
            
            return(self._word_ngrams(stemmed_tokens, stop_words))

        return(analyzer)