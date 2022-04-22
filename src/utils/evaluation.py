import argparse
from typing import Dict, List, Set
from torchmetrics.functional import (
    f1_score,
    hamming_distance,
    precision,
    recall,
    accuracy,
)
import torch
import data_split
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("iso", help="TGT iso to grade")

    args = parser.parse_args()

    iso = args.iso

    if ";" in iso:
        splt_iso = iso.split(";")
        dfs = []
        for iso in splt_iso:
            df = evaluate_all(iso)
            df["iso"] = iso
            dfs.append(df)
        df = pd.concat(dfs)

        print(df)
    else:
        print(evaluate_all(iso))


def load_unimorph(fp: str):
    f = open(fp, "r")
    lns = f.readlines()
    f.close()

    allowed_dimensions = {"Case", "PartOfSpeech", "Person", "Number"}

    fm = data_split.feature_map()
    allowed_fts = [k for k, v in fm.items() if v in allowed_dimensions]
    out: Dict[str, Set[str]] = {}
    for ln in lns:
        splt_ln = ln.split()
        tok = splt_ln[1]
        labels = splt_ln[2:]
        labels = [l for l in labels if l != "none"]

        allowed_labels = [l for l in labels if l in allowed_fts]

        if len(labels) != 0:
            out[tok] = set(allowed_labels)

    return out


def get_fps(iso: str) -> List[str]:
    base = f"./data/unimorph/generated/{iso}."
    src_emb = base + "src_embed_fts"
    tgt_emb = base + "tgt_embed_fts"
    src_token = base + "src_token_fts"
    tgt_token = base + "tgt_token_fts"

    return src_emb, tgt_emb, src_token, tgt_token


def evaluate_all(iso: str):
    fps = get_fps(iso)

    base_fp = f"./data/unimorph/generated/{iso}."
    results = []
    for fp in fps:
        f1, hamming, prec, recall, acc = evaluate(iso, fp)

        kind = fp.replace(base_fp, "")
        out = {
            "model": kind,
            "f1": f1,
            "precision": prec,
            "recall": recall,
            "hamming": hamming,
            "accuracy": acc,
        }
        results.append(out)

    df = pd.DataFrame(results)

    return df


def evaluate(iso, fp):
    test_fp = f"./data/unimorph/test/{iso}"
    test = load_unimorph(test_fp)
    pred = load_unimorph(fp)

    # extract those terms which occur in both
    test_tokens = set(test.keys())
    pred_tokens = set(pred.keys())

    classes = set()
    test_labels = [classes.update(l) for l in test.values()]
    pred_labels = [classes.update(l) for l in pred.values()]
    classes = list(classes)
    num__class = len(classes)

    label2id = {}
    for i, c in enumerate(classes):
        label2id[c] = i

    overlap_terms = test_tokens.intersection(pred_tokens)

    y = torch.zeros(len(overlap_terms), num__class, dtype=int)
    y_hat = torch.zeros(len(overlap_terms), num__class, dtype=int)

    overlap_terms = test_tokens.intersection(pred_tokens)

    for i, term in enumerate(overlap_terms):
        t = test[term]
        p = pred[term]

        for l in t:
            idx = label2id[l]
            y[i, idx] = 1

        for l in p:
            idx = label2id[l]
            y_hat[i, idx] = 1

        t_str = ";".join(t)
        p_str = ";".join(p)

        # print(term + "\t" + t_str + "\t" + p_str)

    f1 = round(f1_score(y_hat, y, multiclass=False).item(), 3)
    ham = round(hamming_distance(y_hat, y).item(), 3)
    prec = round(precision(y_hat, y, multiclass=False).item(), 3)
    r = round(recall(y_hat, y, multiclass=False).item(), 3)
    acc = round(accuracy(y_hat, y, multiclass=False).item(), 3)

    return f1, ham, prec, r, acc


parse_args()
