from typing import List, Optional, Union

class PreProcessor:
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        self.corpus = self.preprocess(corpus) if corpus else []

    def preprocess(self, text: Optional[Union[List[str], str]]) -> Optional[Union[List[str], str]]:
        if isinstance(text, str):
            return text.lower()
        elif isinstance(text, list):
            return [sentence.lower() for sentence in text]
        else:
            return text
