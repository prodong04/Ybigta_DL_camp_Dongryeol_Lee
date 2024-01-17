import re, collections
from typing import Optional, List, Union 
from collections import Counter, defaultdict
from YBIGTA.parent import BaseTokenizer
from YBIGTA.preprocessor import PreProcessor 

class BPETokenizer(BaseTokenizer):
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        super().__init__(corpus)
        self.preprocessor = PreProcessor(corpus)
        self.corpus = self.preprocessor.corpus
        self.split = {}
        self.merge = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.current_id = 0

    def train(self, n_iter: int = None) -> None:
    # BPE 알고리즘에 따라 훈련 데이터에서 철자들을 합치고, 빈도수를 기반으로 토큰을 생성하는 로직
        # 빈도수 계산
        self.token_counts = Counter(self.corpus)

        alphabet = []
        for word in self.token_counts.keys():
            for letter in word:
                if letter not in alphabet:
                    alphabet.append(letter)
        alphabet.sort()


        # add the special token </w> at the beginning of the vocabulary
        vocab = ["</w>"] + alphabet.copy()
        for token in vocab:
            # token_id를 초기화합니다. 
            self.token_to_id[token] = self.current_id
            self.id_to_token[self.current_id] = token
            self.current_id += 1

        # split each word into individual characters before training
        self.splits = {word: [c for c in word] for word in self.corpus}
        while n_iter>0:
            pair_freqs = self.compute_pair_freqs()

            # find the most frequent pair
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # merge the most frequent pair
            self.splits = self.merge_pair(*best_pair)
            self.merge[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
            token = best_pair[0] + best_pair[1]
            self.token_to_id[token] = self.current_id
            self.id_to_token[self.current_id] = token
            self.current_id += 1
            n_iter -= 1
        return self.merge

    def compute_pair_freqs(self):
        """Pair의 빈도수를 계산합니다."""

        pair_freqs = defaultdict(int)
        for word, freq in self.token_counts.items():
            split = self.splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def merge_pair(self, a, b):
        """Merge the given pair."""

        for word in self.token_counts.keys():
            split = self.splits[word]
            if len(split) == 1:
                continue
            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            self.splits[word] = split
            breakpoint()
        return self.splits
    

    def tokenize(self, text: Optional[Union[List[str], str]], 
                 padding: bool = False, max_length: Optional[int] = None):
        if isinstance(text, str):
            tokens = text.split()
        elif isinstance(text, list):
            tokens = [t.split() for t in text]
        else:
            raise ValueError("Input must be either a string or a list of strings.")
        
        # BPE 토크나이저로 텍스트를 토큰화
        tokens_bpe = []

        for sentence_tokens in tokens:
            sentence_tokens_bpe = []

            for word in sentence_tokens:
                # BPE 알고리즘에 따라 토큰화
                tokenized_word = [self.token_to_id.get(char, self.current_id) for char in word]
                sentence_tokens_bpe.extend(tokenized_word)

            tokens_bpe.append(sentence_tokens_bpe)

        # 필요에 따라 패딩 수행
        if padding:
            max_len = max_length or max(len(tokens_bpe) for tokens_bpe in tokens_bpe)
            tokens_bpe = [sentence + [0] * (max_len - len(sentence)) if len(sentence) < max_len else sentence[:max_len] for sentence in tokens_bpe]


        return tokens_bpe
    



class WordTokenizer(BaseTokenizer):
    def __init__(self, corpus: Optional[Union[List[str], str]] = None):
        super().__init__(corpus)
        self.preprocessor = PreProcessor(corpus)
        self.corpus = self.preprocessor.corpus

    def train(self, n_iter) -> None:
        if isinstance(self.corpus, str):
            # Tokenize based on whitespace
            words = self.corpus.split()
            # Remove duplicate words while preserving the order
            unique_words = list(dict.fromkeys(words))
            # Assign token IDs
            self.token_to_id = {word: i for i, word in enumerate(unique_words)}
            self.id_to_token = {i: word for i, word in enumerate(unique_words)}
            self.current_id = 0
        elif isinstance(self.corpus, list):
            # Tokenize each sentence based on whitespace, words에서부터 안들어가있는데...?
            words = [word for sentence in self.corpus for word in sentence.split()]
            # Remove duplicate words while preserving the order
            unique_words = list(dict.fromkeys(words))
            # Assign token IDs
            self.token_to_id = {word: i for i, word in enumerate(unique_words)}
            self.id_to_token = {i: word for i, word in enumerate(unique_words)}
            self.current_id = 0

        else:
            raise ValueError("Input must be either a string or a list of strings.")
        return self.token_to_id

    def tokenize(self, text: Optional[Union[List[str], str]], padding: bool = False, max_length: Optional[int] = None):
        self.preprocessor = PreProcessor(text)
        text = self.preprocessor.corpus
        if not self.token_to_id:
            raise ValueError("Tokenizer has not been trained. Call train method first.")

        if isinstance(text, str):
            words = text.split()
        elif isinstance(text, list):
            words = [word for sentence in text for word in sentence.split()]
        else:
            raise ValueError("Input must be either a string or a list of strings.")

        # Convert words to token ids
        tokens_word = [self.token_to_id.get(word, self.current_id) for word in words]



        # Padding if necessary
        if padding:
            max_len = max_length or len(tokens_word)
            tokens_word = tokens_word[:max_len] + [0] * (max_len - len(tokens_word))

        return tokens_word
