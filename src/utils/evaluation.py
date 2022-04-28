import argparse
from collections import defaultdict
from typing import DefaultDict, Dict, List, Set
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

import numpy as np
from data_split import feature_map
from sklearn.metrics import mutual_info_score as mut_info
from statistics import mean
import lang2vec.lang2vec as l2v
import itertools


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("iso", help="TGT iso to grade")

    args = parser.parse_args()

    iso = args.iso

    if ";" in iso:
        splt_iso = iso.split(";")
        dfs = []
        from tqdm import tqdm

        for iso in tqdm(splt_iso):
            df = evaluate_all(iso)
            src, tgt = iso.split("_")
            df["src_iso"] = src
            df["tgt_iso"] = tgt
            dfs.append(df)
        df = pd.concat(dfs)
        df = df.sort_values(by=["tgt_iso", "f1"], ascending=False)

        df.to_csv("metrics.tsv", sep="\t")
    else:
        print(evaluate_all(iso))


def load_unimorph(
    fp: str,
    fp2=None,
    fp3=None,
    allowed_dimensions={"Case", "PartOfSpeech", "Person", "Number"},
):
    f = open(fp, "r")
    lns = f.readlines()
    if len(lns) == 0:
        return {}
    f.close()
    if fp2 != None:
        f = open(fp2, "r")
        lns2 = f.readlines()
        lns.extend(lns2)
        f.close()

    if fp3 != None:
        f = open(fp3, "r")
        lns2 = f.readlines()
        lns.extend(lns2)
        f.close()

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


def get_fps(isos: str) -> List[str]:
    base = f"./data/unimorph/generated/{isos}."
    src_emb = base + "src_embed_fts"
    tgt_emb = base + "tgt_embed_fts"
    src_token = base + "src_token_fts"
    tgt_token = base + "tgt_token_fts"

    return src_emb, tgt_emb, src_token, tgt_token


def evaluate_all(iso: str):
    fps = list(get_fps(iso))
    fps.extend(list(make_merged(iso, fps)))

    base_fp = f"./data/unimorph/generated/{iso}."
    results = []

    allowed_dimensions = [
        {"Case"},
        {"PartOfSpeech"},
        {"Person"},
        {"Number"},
        {"Case", "PartOfSpeech", "Person", "Number"},
    ]

    cached_sync: Dict[str, float] = {}

    for ad in allowed_dimensions:
        for fp in fps:
            f1, hamming, prec, recall, acc = evaluate(iso, fp, ad)

            tgt_iso = iso.split("_")[1]
            src_iso = iso.split("_")[0]
            dist = 0  # l2v.distance("genetic", src_iso, tgt_iso)
            kind = fp.replace(base_fp, "")
            if tgt_iso not in cached_sync.keys():
                sync_measure = syncretism_measure(iso)
                cached_sync[tgt_iso] = sync_measure

            if len(ad) == 1:
                dim_str = list(ad)[0]
            else:
                dim_str = "all"
            out = {
                "model": kind,
                "f1": f1,
                "precision": prec,
                "recall": recall,
                "hamming": hamming,
                "accuracy": acc,
                "syncretism": cached_sync[tgt_iso],
                "genetic_distance": dist,
                "dimension": dim_str,
            }
            results.append(out)

        f1, hamming, prec, recall, acc = strawman(iso, fp, allowed_dimensions=ad)

        kind = fp.replace(base_fp, "")
        out = {
            "model": "baseline",
            "f1": f1,
            "precision": prec,
            "recall": recall,
            "hamming": hamming,
            "accuracy": acc,
            "dimension": dim_str,
        }
        results.append(out)

    df = pd.DataFrame(results)

    return df


def strawman(iso, fp, allowed_dimensions={"Case", "PartOfSpeech", "Person", "Number"}):
    # compute metrics for some straw-man task
    src_iso = iso.split("_")[0]
    tgt_iso = iso.split("_")[1]
    # obtain the most common label for each category
    fm = feature_map()
    train_fp = f"./data/unimorph/train/{src_iso}"
    f = open(train_fp)
    lns = f.readlines()
    f.close()
    parsed_lns = [[x for x in ln.split() if x != "none"] for ln in lns]

    counts: DefaultDict[str, DefaultDict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for ln in parsed_lns:
        labels = ln[2:]
        for l in labels:
            dim = fm[l]

            counts[dim][l] += 1

    # get max in category
    maxes = []
    for d, counts in counts.items():
        max_d = max(counts, key=counts.get)
        maxes.append(max_d)

    test_fp = f"./data/unimorph/test/{tgt_iso}"
    test_fp_2 = f"./data/unimorph/test/{tgt_iso}"
    valid_fp_2 = f"./data/unimorph/valid/{tgt_iso}"

    try:
        test = load_unimorph(
            test_fp,
            fp2=test_fp_2,
            fp3=valid_fp_2,
            allowed_dimensions=allowed_dimensions,
        )

        pred = load_unimorph(fp, allowed_dimensions=allowed_dimensions)
    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # extract those terms which occur in both
    test_tokens = set(test.keys())
    pred_tokens = set(pred.keys())
    fm = feature_map()
    classes = set()

    for l in test.values():
        for curr_l in list(l):
            if fm[curr_l] in allowed_dimensions:
                classes.add(curr_l)

    for l in pred.values():
        for curr_l in list(l):
            if fm[curr_l] in allowed_dimensions:
                classes.add(curr_l)

    classes = list(classes)
    classes = []

    for k, v in fm.items():
        if v in allowed_dimensions:
            classes.append(k)
    num__class = len(classes)

    label2id = {}
    label2id["UNK"] = 0
    for i, c in enumerate(classes):
        label2id[c] = i + 1

    overlap_terms = test_tokens.intersection(pred_tokens)

    y = torch.zeros(len(overlap_terms), num__class + 1, dtype=int)
    y_hat = torch.zeros(len(overlap_terms), num__class + 1, dtype=int)

    overlap_terms = test_tokens.intersection(pred_tokens)

    p = set(maxes)
    for i, term in enumerate(overlap_terms):
        t = test[term]

        for l in t:
            idx = label2id.get(l, 0)
            y[i, idx] = 1

        for l in p:
            idx = label2id.get(l, 0)
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


def syncretism_measure(isos: str):
    # create a metric for measuring the syncretism of
    # a given iso code
    # In this case, use the target language

    """
    syncretism for one dimension:
        (MI(ft1, ft2) + MI(f1, f3) + MI(f2, f3) ....) / num_fts
    """

    tgt_iso = isos.split("_")[1]
    # obtain the most common label for each category
    fm = feature_map()
    train_fp = f"./data/unimorph/train/{tgt_iso}"
    f = open(train_fp)
    lns = f.readlines()
    f.close()
    parsed_lns = [[x for x in ln.split() if x != "none"] for ln in lns]

    seen_dim = set()
    for ln in parsed_lns:
        labels = ln[2:]
        for l in labels:
            dim = fm[l]
            seen_dim.add(dim)

    # for each dimension, calculate pairwise mutual information

    mean_infos: Dict[str, float] = {}
    for dim in list(seen_dim):
        # make dataset
        if dim == "PartOfSpeech":
            continue
        data = []

        for ln in parsed_lns:
            labels = ln[2:]
            rel_labels = [l for l in labels if fm[l] == dim]
            conv_ln = defaultdict(int)
            for rl in rel_labels:
                conv_ln[rl] = 1

            data.append(conv_ln)

        dim_data = pd.DataFrame(data)
        dim_data = dim_data.replace(np.nan, 0)

        infos = []
        for ft_y in dim_data:
            for other_ft in dim_data:
                if ft_y == other_ft:
                    continue
                x = dim_data[other_ft]
                y = dim_data[ft_y]
                infos.append(mut_info(y, x))
        if len(infos) == 0:
            # don't calculate if every feature is the same
            continue
        mean_info = mean(infos)
        mean_infos[dim] = mean_info

    syncretism_measure = mean(mean_infos.values())
    return syncretism_measure


def make_merged(isos: str, fps: str):

    union_fp = f"./data/unimorph/generated/{isos}.union"
    intersec_fp = f"./data/unimorph/generated/{isos}.intersec"

    data: DefaultDict[str, List[Set[str]]] = defaultdict(list)

    for fp in fps:
        pred = load_unimorph(fp)

        for t, labels in pred.items():
            data[t].append(set(labels))
    # now merge
    union_data: DefaultDict[str, Set[str]] = defaultdict(set)
    intersec_data: DefaultDict[str, Set[str]] = defaultdict(set)

    for t, values in data.items():
        union_data[t] = set.union(*values)
        intersec_data[t] = set.intersection(*values)

    union_f = open(union_fp, "w")
    for token, labels in union_data.items():
        str_labels = " ".join(labels)
        ln = f"UNK\t{token}\t{str_labels}\n"
        union_f.write(ln)
    union_f.close()

    inter_f = open(intersec_fp, "w")
    for token, labels in intersec_data.items():
        if len(labels) == 0:
            labels.add("UNK")
        str_labels = " ".join(labels)
        ln = f"UNK\t{token}\t{str_labels}\n"
        inter_f.write(ln)
    inter_f.close()

    return union_fp, intersec_fp


def evaluate(iso, fp, allowed_dimensions):
    tgt = iso.split("_")[-1]
    test_fp = f"./data/unimorph/test/{tgt}"
    test_fp_2 = f"./data/unimorph/train/{tgt}"
    valid_fp_2 = f"./data/unimorph/valid/{tgt}"

    test = load_unimorph(
        test_fp, fp2=test_fp_2, fp3=valid_fp_2, allowed_dimensions=allowed_dimensions
    )
    pred = load_unimorph(fp, allowed_dimensions=allowed_dimensions)

    # extract those terms which occur in both
    test_tokens = set(test.keys())
    pred_tokens = set(pred.keys())

    classes = set()
    test_labels = [classes.update(l) for l in test.values()]
    pred_labels = [classes.update(l) for l in pred.values()]
    classes = []
    fm = feature_map()
    for k, v in fm.items():
        if v in allowed_dimensions:
            classes.append(k)
    num__class = len(classes)

    label2id = {}
    label2id["UNK"] = 0
    for i, c in enumerate(classes):
        label2id[c] = i + 1

    overlap_terms = test_tokens.intersection(pred_tokens)

    y = torch.zeros(len(overlap_terms), num__class + 1, dtype=int)
    y_hat = torch.zeros(len(overlap_terms), num__class + 1, dtype=int)

    overlap_terms = test_tokens.intersection(pred_tokens)

    for i, term in enumerate(overlap_terms):
        t = test[term]
        p = pred[term]

        for l in t:
            idx = label2id[l]
            y[i, idx] = 1

        for l in p:
            idx = label2id.get(l, 0)
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
