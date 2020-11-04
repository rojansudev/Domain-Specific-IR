from .document import Document

class InvertedIndex(dict):
    """
    Class to generate and hold inverted index. 
    """
    
    def __init__(self, corpus, collection_freq=None):
        """
        Creating posting lists in inverted index for each word
        """

        for document in corpus:
            for word in document.word_freq:
                if word in self:
                    self[word].append(document.doc_id)
                else:
                    self[word] = [document.doc_id]
