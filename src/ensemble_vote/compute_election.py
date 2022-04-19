# Actually take the elections and deploy them
import argparse
from collections import defaultdict
from enum import unique
import math
from typing import DefaultDict, Dict, List, Set, Tuple
import voter
import elections
import lang2vec.lang2vec as l2v
import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tgt_iso")
    parser.add_argument(
        "vote_method", help="One of: plurality, majority, sum_rule, product_rule"
    )
    parser.add_argument("prediction_out_fp", help="fp of where to store predictions")
    args = parser.parse_args()

    election = setup_election(args.tgt_iso, args.vote_method)

    tgt_unimorph_fp = f"./data/unimorph/test/{args.tgt_iso}"
    X, true_y = get_test_tokens(tgt_unimorph_fp)
    pred_y = [election.run_election(x) for x in tqdm(X, desc="Running election")]

    evaluate(X, true_y, pred_y, args.prediction_out_fp)


def evaluate(x_tokens: List[str], true_y: List[str], pred_y: List[str], out_fp: str):
    # same evaluation function as in classifiers.py

    out_f = open(out_fp, "w")
    evaluation: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    out_f.write("TOKEN\tPREDICTION\tTRUTH\n")

    unique_labels = set()
    for true_labels, pred_labels in zip(true_y, pred_y):
        unique_labels.update(true_labels)
        unique_labels.update(pred_labels)

    num_perfect = 0
    n = 0
    for token, true_labels, pred_labels in zip(x_tokens, true_y, pred_y):

        pred_labels = set(pred_labels)
        true_labels = set(true_labels)

        for label in list(unique_labels):
            if label in pred_labels and label not in true_labels:
                evaluation[label]["fp"] += 1
            if label in pred_labels and label in true_labels:
                evaluation[label]["tp"] += 1
            if label not in pred_labels and label in true_labels:
                evaluation[label]["fn"] += 1
            if label not in pred_labels and label not in true_labels:
                evaluation[label]["tn"] += 1

        if pred_labels == true_labels:
            num_perfect += 1
        n += 1

        true_labels = list(true_labels)
        pred_labels = list(pred_labels)

        true_str = ";".join(true_labels)
        pred_str = ";".join(pred_labels)
        out_f.write(f"{token}\t{true_str}\t{pred_str}\n")
    out_f.close()

    line_acc = num_perfect / n
    line_acc = round(line_acc, 3)

    # calculate summary statistics for each class
    print("label\tprecision\trecall\tfalse_pos_rate\tacc\tperfmatch")
    unique_labels = list(unique_labels)
    unique_labels.sort()
    for label, conf_mat in evaluation.items():
        if conf_mat["fp"] + conf_mat["tp"] == 0:
            prec = 0
        else:
            prec = conf_mat["tp"] / (conf_mat["tp"] + conf_mat["fp"])

        if conf_mat["tp"] + conf_mat["fn"] == 0:
            rec = 0
        else:
            rec = conf_mat["tp"] / (conf_mat["tp"] + conf_mat["fn"])

        prec = round(prec, 3)
        rec = round(rec, 3)

        fp_rate = conf_mat["fp"] / (conf_mat["fp"] + conf_mat["tn"])
        fp_rate = round(fp_rate, 3)

        acc = (conf_mat["tp"] + conf_mat["tn"]) / (
            conf_mat["tp"] + conf_mat["fp"] + conf_mat["tn"] + conf_mat["fn"]
        )
        acc = round(acc, 3)

        print(f"{label}\t{prec}\t{rec}\t{fp_rate}\t{acc}\t{line_acc}")


def get_test_tokens(tgt_unimorph_fp: str) -> Tuple[List[str], List[str]]:
    f = open(tgt_unimorph_fp, "r")
    lns = f.readlines()

    x = []
    y = []
    for ln in lns:
        ln = ln.replace("\n", "")
        splt_ln = ln.split("\t")
        token = splt_ln[1]
        labels = splt_ln[2].split()

        x.append(token)
        y.append(labels)

    return x, y


def setup_election(tgt_iso: str, vote_method: str) -> elections.Election:
    # based on input, return the right kind of election

    voters, weights = make_voters_and_weights(tgt_iso)

    if vote_method == "plurality":
        return elections.Plurality(voters)
    elif vote_method == "majority":
        return elections.Majority(voters)
    elif vote_method == "sum_rule":
        return elections.SumRule(voters, weights)
    elif vote_method == "product_rule":
        return elections.ProductRule(voters, weights)
    elif vote_method == "union":
        return elections.UnionVote(voters)
    elif vote_method == "intersect":
        return elections.IntersectVote(voters)
    elif vote_method == "threshold2":
        return elections.ThresholdVote(voters, 2)
    elif vote_method == "threshold3":
        return elections.ThresholdVote(voters, 3)
    elif vote_method == "threshold4":
        return elections.ThresholdVote(voters, 4)


def language_sims(
    src_isos: List[str], tgt_iso: str, method="genetic"
) -> Dict[str, float]:
    # load language similarity weights
    # src_iso --> weight (similarity)

    out: Dict[str, float] = {}
    for src_iso in src_isos:
        dist = l2v.distance(method, src_iso, tgt_iso)
        sim = abs(1 - dist)
        out[src_iso] = sim

    return out


def make_voters_and_weights(tgt_iso: str) -> Tuple[List[voter.Voter], Dict[str, float]]:
    # collect all predictors that were applied to this target ISO
    # return [voters], {weights}
    voters: List[voter.Voter] = []

    seen_isos: Set[str] = set()
    for fp in tqdm(
        glob.glob(f"./data/evaluation/*/{tgt_iso}_test_predictions_tgt.tsv"),
        desc="Making voters",
    ):
        src_iso = (fp.split("/")[-2]).split("_")[0]
        if src_iso == tgt_iso:
            # don't want to give us scores from ourselves
            continue
        v = voter.Voter(src_iso, fp)
        voters.append(v)

        seen_isos.add(src_iso)

    weights = language_sims(list(seen_isos), tgt_iso)

    return voters, weights


if __name__ == "__main__":
    parse_args()
