from typing import List, Optional, Union
from YBIGTA.preprocessor import PreProcessor

class BaseTokenizer:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        self.preprocessor = PreProcessor(corpus)
        self.vocab = {}
        self.corpus = self.preprocessor.corpus

    def add_corpus(self, corpus: Union[List[str], str]) -> None:
        self.preprocessor = PreProcessor(corpus)
        new_corpus = self.preprocessor.corpus
        self.corpus.extend(new_corpus)

    def train(self) -> None:
        raise NotImplementedError("need child class")

    def tokenize(self) -> Union[List[List[int]], List[int]]:
        raise NotImplementedError("need child class")

    def __call__(self, text: Union[List[str], str], padding: bool = False, max_length: Optional[int] = None) -> Union[List[List[int]], List[int]]:
        return self.tokenize(text, padding, max_length)
