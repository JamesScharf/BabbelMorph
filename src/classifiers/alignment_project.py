# Deploy the UniMorph classifier on the train side of the text and save to ./unimorph/data/generated/tgt
from collections import defaultdict
from typing import Counter, DefaultDict, List, Dict, Union, Set

from text_unimorph_classifier import make_classifier, evaluate_classifier
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate potential UniMorph data for language and save in ./unimorph/generated"
    )

    parser.add_argument("--fp", help="The dictionary file")
    parser.add_argument("--src", help="The src iso")
    parser.add_argument("--tgt", help="The tgt iso")
    args = parser.parse_args()

    fp = args.fp
    src = args.src
    tgt = args.tgt

    ad = AlignmentDictionary(src, tgt, fp)
    ad.pipeline(fp)


class AlignmentDictionary(object):
    def __init__(self, src_iso: str, tgt_iso: str, fp: str):
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.ft2dim = self.load_ft2dim()

    def pipeline(self, dict_fp: str):
        d = self.load_dictionary(dict_fp)
        d_w_fts = self.classify(d)
        # self.save_intermediate(d_w_fts)
        merged_d = self.merge_duplicates(d_w_fts)
        self.save_final(merged_d)

        model_save_fp = (
            f"./data/trained_classifiers/{self.src_iso}_{self.tgt_iso}/final.ckpt"
        )
        src_test = evaluate_classifier(model_save_fp, self.src_iso, self.tgt_iso, "src")
        f = open("./data/classifier_training_metrics/src_metrics.txt", "a+")
        f1 = src_test["test_f1"]
        loss = src_test["test_loss"]
        ham = src_test["test_hamming"]
        ln = f"{self.src_iso}\t{self.tgt_iso}\t{f1}\t{ham}\t{loss}\n"
        f.write(ln)
        f.close()

    def load_dictionary(self, fp) -> List[Dict[str, Union[str, Set[str]]]]:
        f = open(fp, "r")
        lns = f.readlines()
        f.close()

        output: List[Dict[str, Union[str, Set[set]]]] = []
        for ln in lns:
            splt_ln = ln.split()
            src_tok, tgt_tok = splt_ln[0], splt_ln[1]

            data = {
                "src_token": src_tok,
                "tgt_token": tgt_tok,
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
        # src_tokens = [("src", x["src_token"]) for x in d]
        # _, str_classifier = make_classifier(self.src_iso, self.tgt_iso, False)
        # textmodel_src_pred_labels = str_classifier.predict(src_tokens)
        # tgt_tokens = [("tgt", x["tgt_token"]) for x in d]
        # textmodel_tgt_pred_labels = str_classifier.predict(tgt_tokens)

        # apply embedding classifier
        src_tokens = [("src", x["src_token"]) for x in d]
        _, classifier = make_classifier(self.src_iso, self.tgt_iso)
        pred_labels_src = classifier.predict(src_tokens)
        tgt_tokens = [("tgt", x["tgt_token"]) for x in d]
        pred_labels_tgt = classifier.predict(tgt_tokens)

        # save results
        out: List[Dict[str, Union[str, Set[str]]]] = []
        for d_entry, p_src, p_tgt in zip(
            d,
            pred_labels_src,
            pred_labels_tgt,
        ):
            d_entry["src_labels"] = p_src
            d_entry["tgt_labels"] = p_tgt

            out.append(d_entry)

        return out

    def merge_duplicates(
        self, d: List[Dict[str, Union[str, Set[str]]]]
    ) -> Dict[str, List[str]]:
        # combine duplicate entries on tgt side through a mechanism
        # possible mechanisms: 1) Take only one label in each category, 2) Take all labels (no vote)

        # must first run classify!!!

        # dimension= src_token_fts tgt_token_fts src_embed_fts tgt_embed_fts
        pred_type = ["src_labels", "tgt_labels"]

        out: Dict[str, Dict[str, Union[str, Set[str]]]] = {}
        for i, d_entry in enumerate(d):

            tgt_tok = d_entry["tgt_token"]
            out[tgt_tok] = {"tgt_token": tgt_tok}
            for pt in pred_type:
                all_preds = d_entry[pt]

                # find the highest votes by category
                counter: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
                    lambda: defaultdict(int)
                )
                for ft in all_preds:
                    dim = self.ft2dim.get(ft, "none")
                    if dim != "none":
                        counter[dim][ft] += 1

                remaining = set()

                for dim, candidates in counter.items():
                    remaining.update(set(candidates))

                out[tgt_tok][pt] = remaining

        return list(out.values())

    def save_intermediate(self, d: List[Dict[str, Union[str, Set[str]]]]):
        # save the intermediate data to ./data/dictionary_applications/src_tgt.tsv

        pred_type = ["src_token_fts", "tgt_token_fts", "src_embed_fts", "tgt_embed_fts"]
        fp = f"./data/dictionary_applications/{self.src_iso}_{self.tgt_iso}"

        out_f = open(fp, "w")
        header = "src_tok_fts\ttgt_tok_fts\tsrc_emb_fts\ttgt_emb_fts\n"
        out_f.write(header)

        for i, d_entry in enumerate(d):
            stf = "+".join(d_entry["src_token_fts"])
            ttf = "+".join(d_entry["tgt_token_fts"])
            sef = "+".join(d_entry["src_embed_fts"])
            tef = "+".join(d_entry["tgt_embed_fts"])

            src = d_entry["src_token"]
            tgt = d_entry["tgt_token"]

            ln = f"{src}\t{tgt}\t{stf}\t{ttf}\t{sef}\t{tef}\n"
            out_f.write(ln)

        out_f.close()

    def save_final(self, d: List[Dict[str, Union[str, Set[str]]]]):
        # save the data to ./unimorph/data/generated/tgt
        # in UniMorph-format

        pred_type = ["src_labels", "tgt_labels"]

        for pt in pred_type:
            fp = f"./data/unimorph/generated/{self.src_iso}_{self.tgt_iso}.{pt}"

            out_f = open(fp, "w")

            for i, d_entry in enumerate(d):
                fts = list(d_entry[pt])
                fts.sort()
                fts = " ".join(fts)
                tgt = d_entry["tgt_token"]
                lemma = "UNK"

                ln = f"{lemma}\t{tgt}\t{fts}\n"
                out_f.write(ln)

            out_f.close()


if __name__ == "__main__":
    parse_args()
