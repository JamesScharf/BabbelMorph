# In the process of building our embeddings, we drop
# some items
# This should help reconstruct them
from cgitb import lookup
from collections import defaultdict
from glob import glob
from re import L
import numpy as np
from typing import DefaultDict, Dict, List, Tuple, final
import morfessor
import sys

sys.path.insert(1, "./src/utils")
from tqdm import tqdm
import morfessor_utils as mu
from sklearn.neighbors import NearestNeighbors
import ray
import torch


def load_bilingual_embed(fp: str) -> Dict[str, List[float]]:

    f = open(fp, "r")
    lns = f.readlines()

    def parse_ln(ln: str) -> Tuple[str, List[float]]:
        splt_ln = ln.split()
        key = splt_ln[0]
        str_values = splt_ln[1:]

        values = []
        for v in str_values:
            try:
                f_v = float(v)
                values.append(f_v)
            except:
                f_v = 0.0
                values.append(f_v)

        return (key, values)

    parsed_lns = [parse_ln(ln) for ln in lns]

    # merge into dictionary
    lookup_table: DefaultDict[str, List[float]] = defaultdict(lambda: [0] * 50)

    for p_ln in parsed_lns:
        lookup_table[p_ln[0]] = torch.tensor(p_ln[1])

    return lookup_table


def load_morfessor_model(model_fp: str):
    io = morfessor.MorfessorIO()
    model = io.read_any_model(model_fp)
    return model


def get_token_embedding(
    morf_model,
    lookup_table: Dict[str, List[float]],
    token: str,
    segment_method: str,
    skip_segment=False,
) -> List[float]:

    if skip_segment == True:
        segmented_tok = token
    else:
        segmented_tok = mu.segment_token(morf_model, token, segment_method)
    # obtain each embedding
    embeddings = []
    for morpheme in segmented_tok.split():
        embedding = lookup_table.get(morpheme, None)
        if embedding != None:
            embeddings.append(embedding)

    if len(embeddings) >= 1:
        embeds = torch.stack(embeddings)
        centroid = embeds.mean(dim=0)
        suffix_embedding = centroid.tolist()
        return suffix_embedding
    else:
        return None


def make_token_vector_data(
    suffix_embed_fp: str, morf_model_fp: str, token_lst: List[str]
) -> Dict[str, List[float]]:

    morf_model = load_morfessor_model(morf_model_fp)
    suffix_lookup_table = load_bilingual_embed(suffix_embed_fp)

    token_lookup_table = [
        (t, get_token_embedding(morf_model, suffix_lookup_table, t)) for t in token_lst
    ]

    # lint out nones
    token_lookup_table = [x for x in token_lookup_table if x[1] != None]

    out_dict: Dict[str, List[float]] = {}

    for t, vec in token_lookup_table:
        out_dict[t] = vec

    return out_dict


def nearest_tgt_token(
    src_token_lookup: Dict[str, List[float]],
    tgt_token_lookup: Dict[str, List[float]],
    tgt_tokens: List[str],
    k=10,
) -> Dict[str, List[str]]:

    # given some target language token list, find the nearest tokens in the source language
    src_tokens = []
    src_values = []

    for k, v in src_token_lookup.items():
        src_tokens.append(k)
        src_values.append(v)

    tgt_values = [tgt_token_lookup[x] for x in tgt_tokens]

    nbrs = NearestNeighbors(n_neighbors=k).fit(src_values)

    indices = nbrs.kneighbors(tgt_values, return_distance=False)

    # recover indices
    output: List[str] = []

    for tgt_tok, curr_indices in zip(tgt_tokens, indices):
        tokens = [(tgt_tok, src_tokens[i]) for i in curr_indices]

        print(tgt_tok)
        print(src_tokens)
        print()
        output.append((tgt_tok, tokens))

    return output


def get_bible_tokens(iso: str):

    print(iso)
    toks = set()
    for fp in glob(f"./data/bible_corpus/{iso}*"):
        f = open(fp)
        lns = f.readlines()
        tokens = [x.split() for x in lns]

        final_toks = []
        for ln_toks in tokens:
            final_toks.extend(ln_toks)
        toks.update(final_toks)

    return list(set(toks))


def parse_args():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("suffix_embedding_fp")
    parser.add_argument("iso")  # iso code so we can find text
    parser.add_argument("morfessor_model_fp")

    args = parser.parse_args()

    tokens = get_bible_tokens(args.iso)
    token_lookup = make_token_vector_data(
        args.suffix_embedding_fp, args.morfessor_model_fp, tokens
    )


if __name__ == "__main__":
    parse_args()
