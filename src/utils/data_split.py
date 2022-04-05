from collections import defaultdict
from email.policy import default
from enum import unique
from genericpath import exists
from random import random
from typing import Tuple, List, Set, DefaultDict
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import argparse
import random
from tqdm import tqdm
import morfessor as morf
import numpy as np
import numpy as np
import morfessor_utils as mu


def combine_duplicates(in_fp: str, out_fp: str, morfessor_model):
    """Merge together duplicate word forms

    Args:
        in_fp (str): Original format raw UniMorph
        out_fp (str): A TSV file
    """

    f = open(in_fp, "r")

    pairs: DefaultDict[Tuple[str, str], Set[str]] = defaultdict(set)
    for ln in tqdm(f):
        if len(ln.split()) == 0:
            continue

        splt_ln = ln.split()

        lemma = splt_ln[0]
        token = splt_ln[1]
        labels = splt_ln[2].split(";")

        new_labels = []
        for l in labels:
            if l.upper() != l:
                continue
            new_labels.extend(l.split("+"))
        new_labels.sort()
        unique_labels = set(new_labels)

        pairs[(lemma, token)].update(unique_labels)

    f.close()

    # now binarize
    tokens: List[Tuple[str, str]] = []
    labels: List[List[str]] = []

    for k, l in pairs.items():
        # lint out fakes
        lemma, token = k
        tokens.append((lemma, token))
        labels.append(list(l))

    # make new file
    import morfessor

    io = morfessor.MorfessorIO()
    model = io.read_any_model(morfessor_model)
    out_f = open(out_fp, "w")
    for token, label in zip(tokens, labels):
        new_labels = [f"__label__{l}" for l in label]
        new_labels.sort()
        str_lab = " ".join(new_labels)
        token = token[1]
        t = mu.segment_token(model, token)
        ln = f"{str_lab}\t{t}\n"
        out_f.write(ln)

    out_f.close()


def data_split(
    combo_fp: str,
    train_out_fp: str,
    valid_out_fp: str,
    test_out_fp: str,
) -> Tuple[List[str], List[str]]:

    combo_f = open(combo_fp, "r")
    train_f = open(train_out_fp, "w")
    valid_f = open(valid_out_fp, "w")
    test_f = open(test_out_fp, "w")

    i = 0

    train_lemma = set()
    test_lemma = set()
    valid_lemma = set()

    # set random seed
    random.seed(42)
    for ln in tqdm(combo_f, desc="Data split"):
        if i == 0:
            header = ln
            train_f.write(header)
            test_f.write(header)
            valid_f.write(header)
        else:

            lemma = ln.split("\t")[0]
            r = random.uniform(0, 1)
            if lemma in train_lemma:
                train_f.write(ln)
            elif lemma in valid_lemma:
                valid_f.write(ln)
            elif lemma in valid_lemma:
                test_f.write(ln)
            elif r < 0.7:  # Train
                train_f.write(ln)
                train_lemma.add(lemma)
            elif r >= 0.7 and r <= 0.8:  # valid
                valid_f.write(ln)
                valid_lemma.add(lemma)
            else:
                test_f.write(ln)
                test_lemma.add(lemma)

        i += 1

    train_f.close()
    test_f.close()
    valid_f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_fp", help="Raw UniMorph data")
    parser.add_argument(
        "squashed_fp", help="Where to store the squashed version of raw_fp"
    )
    parser.add_argument("train_fp", help="")

    parser.add_argument("valid_fp", help="")

    parser.add_argument("test_fp", help="")

    parser.add_argument("morfessor_model", help="")

    args = parser.parse_args()

    combine_duplicates(args.raw_fp, args.squashed_fp, args.morfessor_model)
    data_split(
        args.squashed_fp,
        args.train_fp,
        args.valid_fp,
        args.test_fp,
    )

    # make data format


if __name__ == "__main__":
    parse_args()
