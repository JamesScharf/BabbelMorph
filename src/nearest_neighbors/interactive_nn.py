# An interactive test script for nearest neighbors

# Two ways to run:

# python3 ./src/nearest_neighbors/interactive_nn.py ./data/crosslingual_token_embeddings/lat_spa/lat ./data/crosslingual_token_embeddings/lat_spa/spa src
# python3 ./src/nearest_neighbors/interactive_nn.py ./data/crosslingual_token_embeddings/lat_spa/lat ./data/crosslingual_token_embeddings/lat_spa/spa tgt

import sys
from typing import List
import reconstruct_token_vec as rtc

sys.path.insert(1, "./src/utils")
import morfessor_utils as mu
import argparse
from scipy import spatial
import ray


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("src_vec_fp")
    parser.add_argument("tgt_vec_fp")
    parser.add_argument(
        "src_or_tgt_query",
        help="If src then obtain a list of src words from some tgt word; if tgt opposite",
    )

    args = parser.parse_args()

    if args.src_or_tgt_query == "src":
        src_lookup = rtc.load_bilingual_embed(args.src_vec_fp)
        tgt_lookup = rtc.load_bilingual_embed(args.tgt_vec_fp)
    else:
        src_lookup = rtc.load_bilingual_embed(args.tgt_vec_fp)
        tgt_lookup = rtc.load_bilingual_embed(args.src_vec_fp)

    from random import sample

    tgt_word_sample = sample(list(tgt_lookup.keys()), 100)
    print("A sample of target words: ")
    for j in range(0, len(tgt_word_sample), 7):
        print()
        for i in range(0, 7):
            if j + i < len(tgt_word_sample):
                tgt_w = tgt_word_sample[j + i]
                print(f"\t{tgt_w}", end=" ")
    print()

    while True:
        print("-------------------------------")
        tgt_tok = input("Enter a target word:")
        if tgt_tok not in tgt_lookup.keys():
            print("Token not in vocabulary")
            continue
        res = src_query(src_lookup, tgt_lookup, tgt_tok)
        for res_token, res_sim in res:
            res_sim = round(res_sim, 4)
            print(f"\t{res_sim}\t{res_token}")
        print()


@ray.remote
def cosine_similarity(token, vec1, vec2) -> float:
    if len(vec1) != len(vec2):
        return (token, 0)
    sim = 1 - abs(spatial.distance.cosine(vec1, vec2))
    return (token, sim)


def src_query(src_lookup, tgt_lookup, tgt_token: str, k=20):
    # find k nearest src neighbors to some target token
    tgt_tok_embeds = tgt_lookup[tgt_token]

    sims = ray.get(
        [
            cosine_similarity.remote(src_tok, tgt_tok_embeds, src_vec)
            for src_tok, src_vec in src_lookup.items()
        ]
    )

    # rank by similarity
    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[0:k]
    return top_k


if __name__ == "__main__":
    ray.init()
    parse_args()
