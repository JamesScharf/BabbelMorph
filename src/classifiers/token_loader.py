from enum import unique
from typing import List, Dict
import torch
from unidecode import unidecode


# convert tokens to tensors
class TokenLoader:
    def __init__(self, iso: str, to_ascii=True):
        self.to_ascii = to_ascii
        train_tokens = self.get_train_tokens(iso)
        self.char2index = self._make_vocab_mapper(train_tokens)

    def get_train_tokens(self, iso: str) -> List[str]:
        fp = f"./data/unimorph/train/{iso}"
        f = open(fp, "r")
        lns = f.readlines()
        f.close()

        tokens = [ln.split()[1] for ln in lns]
        return tokens

    def _make_vocab_mapper(self, tokens: List[str]) -> Dict[str, int]:
        # make dictionary for index lookup

        unique_chars = set()
        for w in tokens:
            if self.to_ascii:
                w = unidecode(w)
            for c in w:
                unique_chars.add(c)

        lst_chars = list(unique_chars)
        char2index: Dict[str, int] = {}

        char2index["UNK"] = 0
        for i, c in enumerate(lst_chars):
            char2index[c] = i + 1
        # add extra for UNK
        return char2index

    def tokens2tensor(self, tokens: List[str], max_len=16) -> torch.tensor:
        # returns tokens in format needed for pytorch embedding layer
        # only function that the user should be using

        index_lst: List[torch.Tensor] = []

        for w in tokens:
            vectorized: List[int] = []

            if self.to_ascii:
                w = unidecode(w)

            for i, c in enumerate(w):
                # don't allow tokens longer than max_len
                if i < max_len:
                    i = self.char2index.get(c, 0)
                    vectorized.append(i)
            # pad sequence
            padding = [0] * (max_len - len(vectorized))
            vectorized.extend(padding)
            t = torch.tensor(vectorized)
            index_lst.append(t)

        out = torch.vstack(index_lst)

        return out
