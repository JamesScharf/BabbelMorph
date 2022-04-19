# An object that corresponds to one model's predictions for each token

from tkinter import W
from typing import Dict, Set, DefaultDict
from collections import defaultdict


class Voter(object):
    def __init__(self, src_iso: str, tgt_test_fp: str):
        self.src_iso = src_iso
        self.test_fp = tgt_test_fp

        self.legal_fts = self.determine_legal_features()
        self.feature_map = self.ft_to_ix(self.legal_fts)
        self.prediction_map = self.load_predictions(self.feature_map)

    def ft_to_ix(self, legal_features: Set[str]) -> Dict[str, int]:
        fts = list(legal_features)
        out_dict: Dict[str, int] = {}
        for i, ft in enumerate(fts):
            out_dict[ft] = i

        return out_dict

    def determine_legal_features(self):
        # obtain a set of the features that this model can vote on
        # use the train data
        legal_features: Set[str] = set()

        fp = f"./data/unimorph/train/{self.src_iso}"

        f = open(fp, "r")
        lns = f.readlines()
        for ln in lns:
            splt_ln = ln.split("\t")
            fts = splt_ln[2].split()
            legal_features.update(fts)

        return legal_features

    def load_predictions(self, ft_to_ix: Dict[str, int]):
        f = open(self.test_fp, "r")
        lns = f.readlines()
        f.close()

        data: DefaultDict[str, Set[str]] = defaultdict(set)

        for ln in lns:
            ln = ln.replace("TOKEN\tTRUTH\tPREDICTION", "")
            ln = ln.replace("\n", "")
            token, _, preds = ln.split("\t")
            pred_lst = preds.split(";")
            if len(pred_lst) == 1 and pred_lst[0] == "":
                converted = []
            else:
                converted = [ft_to_ix[ft] for ft in pred_lst]

            data[token].update(converted)

        return data

    def vote(self, tgt_token: str, ft: str) -> int:
        # given some token, feature, ask model if vote is:
        #   -1 --- abstain (when the language doesn't support this feature)
        #   0 --- model vote against; *doesn't* have feature
        #   1 --- vote in favor: model *has* this feature

        v = None

        if tgt_token not in self.prediction_map.keys() or ft not in self.legal_fts:
            v = -1
        elif self.feature_map[ft] in self.prediction_map[tgt_token]:
            v = 1
        else:
            v = 0
        return v
