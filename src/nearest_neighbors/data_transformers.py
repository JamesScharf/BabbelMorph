from collections import defaultdict
import reconstruct_token_vec as rtc
from typing import DefaultDict, List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
import sys

import argparse
from tqdm import tqdm
import morfessor_utils as mu


class UniMorphEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        src_morf_model_fp: str,
        tgt_morf_model_fp: str,
        src_suffix_bilingual_embed_fp: str,
        tgt_suffix_bilingual_embed_fp: str,
        segment_method: str,
        dictionary_mode=False,  # if true, then use the source language's token only and lookup the aligned suffix, given some target.
        dictionary_fp="",  # file path to SUFFIX/segmented dictionary
    ):
        self.dictionary_fp = dictionary_fp
        self.dictionary_mode = dictionary_mode

        if dictionary_mode:
            self.tgt_to_src_vocab = self.load_dictionary()
        self.segment_method = segment_method
        self.src_morf = rtc.load_morfessor_model(src_morf_model_fp)
        self.tgt_morf = rtc.load_morfessor_model(tgt_morf_model_fp)
        self.src_suffix_feature_lookup_table = rtc.load_bilingual_embed(
            src_suffix_bilingual_embed_fp
        )
        self.tgt_suffix_feature_lookup_table = rtc.load_bilingual_embed(
            tgt_suffix_bilingual_embed_fp
        )

    def load_dictionary(self) -> Dict[str, List[str]]:
        # tgt-->src suffix dictionary
        f = open(self.dictionary_fp, "r")
        lns = f.readlines()
        out_dict: DefaultDict[str, List[str]] = defaultdict(list)

        for ln in lns:
            src, tgt = ln.split()
            out_dict[tgt].append(src)

        return out_dict

    def convert_tgt_to_src_segmented(self, tgt_token) -> str:
        # we're going to pool the suffix results here
        # each result has multiple possible translations

        tgt_segmented = mu.segment_token(self.tgt_morf, tgt_token, self.segment_method)
        tgt_segments = tgt_segmented.split()

        pos_trans = []
        for seg in tgt_segments:
            tgt_as_src_segmented = self.tgt_to_src_vocab[seg]
            pos_trans.extend(tgt_as_src_segmented)

        pos_trans = list(set(pos_trans))

        pos_trans_str = " ".join(pos_trans)

        return pos_trans_str

    def get_features(self, token, src_or_tgt="src") -> List[float]:
        # get list of vectors given some token
        # if token is src language then src_or_tgt="src" or if src_or_tgt="tgt"
        # then assume token is in target language

        if src_or_tgt == "src" or self.dictionary_mode:

            if self.dictionary_mode and src_or_tgt == "tgt":
                # convert tgt token to src segmented
                src_dict_lookup_res = self.convert_tgt_to_src_segmented(token)
                return rtc.get_token_embedding(
                    self.src_morf,
                    self.src_suffix_feature_lookup_table,
                    src_dict_lookup_res,
                    self.segment_method,
                    skip_segment=True,
                )
            else:
                return rtc.get_token_embedding(
                    self.src_morf,
                    self.src_suffix_feature_lookup_table,
                    token,
                    self.segment_method,
                )

        elif src_or_tgt == "tgt":
            return rtc.get_token_embedding(
                self.tgt_morf,
                self.tgt_suffix_feature_lookup_table,
                token,
                self.segment_method,
            )
        else:
            raise NameError("src_or_tgt should be src or tgt")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # expect that the data has come in the format of:
        # (src, token), (src, token), (tgt, token), (src, token)
        # new_data = Parallel(n_jobs=-1)(
        #    delayed(self.get_features)(t, src_or_tgt=src_or_tgt) for src_or_tgt, t in X
        # )

        new_data = [
            self.get_features(t, src_or_tgt=src_or_tgt)
            for src_or_tgt, t in tqdm(X, desc="Transforming tokens to morphemes...")
        ]

        return new_data


"""
def parse_args():
    # this function only exists for testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("src_suffix_embedding_fp")
    parser.add_argument("tgt_suffix_embedding_fp")

    parser.add_argument("src_morfessor_model_fp")
    parser.add_argument("tgt_morfessor_model_fp")

    args = parser.parse_args()

    trans = UniMorphEmbeddingTransformer(
        args.src_morfessor_model_fp,
        args.tgt_morfessor_model_fp,
        args.src_suffix_embedding_fp,
        args.tgt_suffix_embedding_fp,
    )

    X = [("src", "audire"), ("tgt", "comere")]
    trans.fit(X)
    print(trans.transform(X))


if __name__ == "__main__":
    parse_args()

"""
