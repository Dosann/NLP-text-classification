import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import RegexpStemmer

class StemmedTfidfVectorizer(TfidfVectorizer):
    """TfidVectorizer with RegexpStemmer."""
   
    def build_analyzer(self):
        regexp = RegexpStemmer('ing$|s$|ed$|able$', min=4)
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc:([regexp.stem(word) for word in analyzer(doc)])