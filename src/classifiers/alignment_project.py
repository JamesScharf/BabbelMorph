# Deploy the UniMorph classifier on the train side of the text and save to ./unimorph/data/generated/tgt
from collections import defaultdict
from typing import Counter, DefaultDict, List, Dict, Union, Set

from text_unimorph_classifier import make_classifier


def parse_args():
    pass


class AlignmentDictionary(object):
    def __init__(self, src_iso: str, tgt_iso: str, fp: str):
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.ft2dim = self.load_ft2dim()

    def load_dictionary(self, fp) -> List[Dict[str, Union[str, Set[str]]]]:
        f = open(fp, "r")
        lns = f.readlines()
        f.close()

        output: List[Dict[str, Union[str, Set[set]]]] = []
        for ln in lns:
            splt_ln = ln.split()
            src_tok, tgt_tok = splt_ln[0], splt_ln[1]

            data = {
                "src": src_tok,
                "tgt": tgt_tok,
                "final_fts": set(),  # features determine by consensus
                "src_embed_fts": set(),  # feature determined by application of embed predictor to src side
                "tgt_embed_fts": set(),  # features determined by application of embed predictor to tgt side
                "src_token_fts": set(),  # features determined by application of token predictor to src side
                "tgt_token_fts": set(),  # features determine by application of token predictor to tgt side
            }
            output.append(data)

        return output

    def load_ft2dim(self) -> Dict[str, str]:
        fp = "./data/feature_map.tsv"
        f = open(fp, "r")
        lns = f.readlines()
        f.close()

        ft2dim: Dict[str, str] = {}

        for ln in lns:
            ln = ln.split("\t")
            ft = ln[0]
            dim = ln[1]
            ft2dim[ft] = dim

        return ft2dim

    def classify(
        self, d: List[Dict[str, Union[str, Set[str]]]]
    ) -> List[Dict[str, Union[str, Set[str]]]]:
        # produce dictionary, but with counter as well

        # build classifiers

        # start out with str classifier
        src_tokens = [("src", x["src_token"]) for x in d]
        str_classifier = make_classifier(self.src_iso, self.tgt_iso, False, False)
        textmodel_src_pred_labels = str_classifier.predict(src_tokens)
        tgt_tokens = [("tgt", x["tgt_token"]) for x in d]
        textmodel_tgt_pred_labels = str_classifier.predict(tgt_tokens)

        # apply embedding classifier
        src_tokens = [("src", x["src_token"]) for x in d]
        str_classifier = make_classifier(self.src_iso, self.tgt_iso, False, False)
        embed_src_pred_labels = str_classifier.predict(src_tokens)
        tgt_tokens = [("tgt", x["tgt_token"]) for x in d]
        embed_tgt_pred_labels = str_classifier.predict(tgt_tokens)

        # save results
        out: List[Dict[str, Union[str, Set[str]]]] = []
        for d_entry, txtmod_src, txtmod_tgt, embed_src, embed_tgt in zip(
            textmodel_src_pred_labels,
            textmodel_tgt_pred_labels,
            embed_src_pred_labels,
            embed_tgt_pred_labels,
        ):
            d_entry["src_token_fts"] = txtmod_src
            d_entry["tgt_token_fts"] = txtmod_tgt
            d_entry["src_embed_fts"] = embed_src
            d_entry["tgt_embed_fts"] = embed_tgt

            out.append(d_entry)

        return out

    def merge_duplicates(
        self, d: List[Dict[str, Union[str, Set[str]]]]
    ) -> Dict[str, List[str]]:
        # combine duplicate entries on tgt side through a mechanism
        # possible mechanisms: 1) Take only one label in each category, 2) Take all labels (no vote)

        # must first run classify!!!

        for i, d_entry in enumerate(d):

            all_preds = (
                list(d_entry["src_token_fts"])
                + list(d_entry["tgt_token_fts"])
                + list(d_entry["src_embed_fts"])
                + list(d_entry["tgt_embed_fts"])
            )

            # find the highest votes by category

            counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
                lambda: defaultdict(int)
            )
            for ft in all_preds:
                dim = self.ft2dim[ft]
                counter[dim][ft] += 1

            remaining = set()

            for dim, candidates in counter.items():
                most_votes = max(candidates, key=candidates.get)
                remaining.add(most_votes)

            d[i]["final_fts"] = remaining

        return d

    def save_intermediate(self, d: List[Dict[str, Union[str, Set[str]]]]):
        # save the intermediate data to ./data/dictionary_applications/src_tgt.tsv

        fp = f"./data/dictionary_applications/{self.src}_{self.tgt}"
        pass

    def save_final(self, d: List[Dict[str, Union[str, Set[str]]]]):
        # save the data to ./unimorph/data/generated/tgt
        # in UniMorph-format
        pass
