import reconstruct_token_vec as rtc
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
import sys

import morfessor_utils as mu
import argparse
from joblib import Parallel, delayed


class UniMorphEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        src_morf_model_fp: str,
        tgt_morf_model_fp: str,
        src_bilingual_embed_fp: str,
        tgt_bilingual_embed_fp: str,
    ):
        self.src_morf = rtc.load_morfessor_model(src_morf_model_fp)
        self.tgt_morf = rtc.load_morfessor_model(tgt_morf_model_fp)
        self.src_suffix_feature_lookup_table = rtc.load_bilingual_embed(
            src_bilingual_embed_fp
        )
        self.tgt_suffix_feature_lookup_table = rtc.load_bilingual_embed(
            tgt_bilingual_embed_fp
        )

    def get_features(self, token, src_or_tgt="src") -> List[float]:
        # get list of vectors given some token
        # if token is src language then src_or_tgt="src" or if src_or_tgt="tgt"
        # then assume token is in target language

        if src_or_tgt == "src":
            return rtc.get_token_embedding(
                self.src_morf, self.src_suffix_feature_lookup_table, token
            )

        elif src_or_tgt == "tgt":
            return rtc.get_token_embedding(
                self.tgt_morf, self.tgt_suffix_feature_lookup_table, token
            )
        else:
            raise NameError("src_or_tgt should be src or tgt")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # expect that the data has come in the format of:
        # (src, token), (src, token), (tgt, token), (src, token)
        new_data = Parallel(n_jobs=-1)(
            delayed(self.get_features)(t, src_or_tgt=src_or_tgt) for src_or_tgt, t in X
        )

        return new_data


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
