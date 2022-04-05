import argparse
from collections import defaultdict
from typing import DefaultDict, Dict
import editdistance as ed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Obtain a dictionary from our word alignments. Copied from: https://github.com/tmtmaj/Extract-dictionary-from-bilingual-alignment/blob/main/ExtractTopNDict.py"
    )

    parser.add_argument("--align_fp", help="The file path to the alignment file")
    parser.add_argument("--bitext_fp", help="The file path to the bitext")

    args = parser.parse_args()

    align_fp = args.align_fp
    bitext_fp = args.bitext_fp

    make_dictionary(align_fp, bitext_fp)


def make_dictionary(align_fp, bitext_fp):
    align_f = open(align_fp, "r")
    bitext_f = open(bitext_fp, "r")

    # maintain dictionary
    dictionary: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for align_ln, bitext_ln in zip(align_f, bitext_f):
        src_txt, tgt_txt = bitext_ln.replace("\n", "").split("|||")
        tok_src = src_txt.split()
        tok_tgt = tgt_txt.split()
        aligns_txt_lst = align_ln.replace("\n", "").split(" ")
        txt_aligns = [x.split("-") for x in aligns_txt_lst]
        if len(txt_aligns) == 1:
            continue
        parsed_aligns = [(int(x[0]), int(x[1])) for x in txt_aligns]

        for src_index, tgt_index in parsed_aligns:
            src_w = tok_src[src_index].lower()
            tgt_w = tok_tgt[tgt_index].lower()

            dictionary[src_w][tgt_w] += 1

    passing = 0
    for k, v in dictionary.items():
        for tgt_w, count in v.items():
            editdist = ed.eval(k, tgt_w)
            if editdist < 3 and len(src_w) >= 3 and len(tgt_w) >= 3:  # arbitrary value
                passing += 1
                print(f"{k}\t{tgt_w}")


if __name__ == "__main__":
    parse_args()
