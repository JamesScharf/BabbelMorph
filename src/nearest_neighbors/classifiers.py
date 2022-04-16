from collections import defaultdict
import enum
from lib2to3.pgen2 import token
import data_transformers as dt
from typing import DefaultDict, Dict, List, Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
import argparse
import numpy as np


class UniMorphCB(object):
    def __init__(
        self,
        src_morf_model_fp: str,
        tgt_morf_model_fp: str,
        src_bilingual_embed_fp: str,
        tgt_bilingual_embed_fp: str,
        segment_method: str,
    ):

        self.segment_method = segment_method
        self.embedder = dt.UniMorphEmbeddingTransformer(
            src_morf_model_fp,
            tgt_morf_model_fp,
            src_bilingual_embed_fp,
            tgt_bilingual_embed_fp,
            self.segment_method,
        )
        self.mlb = None

    def process_labels(
        self, unimorph_labels: List[List[str]], train=False
    ) -> List[List[int]]:
        # convert N NOM SG --> one hot encoded

        if self.mlb == None:
            self.mlb = MultiLabelBinarizer()
            proc_labels = self.mlb.fit_transform(unimorph_labels)

        else:
            proc_labels = self.mlb.transform(unimorph_labels)

        return proc_labels

    def load_unimorph(
        self, fp: str, src_or_tgt="src"
    ) -> List[Tuple[List[Tuple[str, str]], List[str]]]:
        # load unimorph train or test file, returning X, Y
        # src_or_tgt should be "src" or "tgt" depending on source
        f = open(fp, "r")
        lns = f.readlines()
        lns = [x.replace("\n", "") for x in lns]

        tokenized = [x.split() for x in lns if len(x.split()) >= 3]
        tokens = [(src_or_tgt, x[1]) for x in tokenized]
        labels = [x[2:] for x in tokenized]

        return tokens, labels

    def train(
        self,
        src_train_tokens: List[Tuple[str]],
        unimorph_labels: List[List[str]],
        src_valid_tokens: List[Tuple[str]],
        valid_unimorph_labels: List[List[str]],
    ) -> None:
        # src_train_tokens looks something like:
        # [(src, token), (src, token), (src, token)]
        # unimorph_labels looks like: [N NOM SG, GEN N SG, N SG, VOC]

        proc_labels = self.process_labels(unimorph_labels, train=True)
        valid_labels = self.process_labels(valid_unimorph_labels)

        self.embedder = self.embedder.fit(src_train_tokens)
        X = self.embedder.transform(src_train_tokens)
        valid_X = self.embedder.transform(src_valid_tokens)

        # remove errors
        new_X = []
        new_y = []

        # establish number of features
        for x in X:
            if x != None:
                self.num_fts = len(x)

        for x, y in zip(X, proc_labels):
            if x == None:
                x = [0] * self.num_fts
            new_X.append(x)
            new_y.append(y)

        # remove errors
        new_X_valid = []
        new_y_valid = []

        for x, y in zip(valid_X, valid_labels):
            if x == None:
                x = [0] * self.num_fts
            new_X_valid.append(x)
            new_y_valid.append(y)

        base = CatBoostClassifier(iterations=200)  # , task_type="GPU", devices="0:1"

        # order doesn't matter except we want POS first
        pos = ["N", "ADJ", "V"]

        order = []
        for p in pos:
            if p in list(self.mlb.classes_):
                p_index = list(self.mlb.classes_).index(p)
                order.append(p_index)

        # nothing else matters
        for i, _ in enumerate(list(self.mlb.classes_)):
            if i not in order:
                order.append(i)

        self.clf = ClassifierChain(base, order=order)

        new_X.append([0] * len(new_X[0]))
        new_y.append([0] * len(self.mlb.classes_))
        self.clf.fit(new_X, new_y)

    def predict(self, tokens: List[Tuple[str]], out_fp: str):
        # tokens: Formatted like (src, token), (tgt, token), (src, token), (src, token)
        # Saves predictions to file like: TOKEN PREDS

        X = self.embedder.transform(tokens)
        # remove errors
        new_X = []
        for x in X:
            if x == None:
                x = [0] * self.num_fts
            new_X.append(x)
        preds = self.clf.predict(new_X)
        preds = self.mlb.inverse_transform(preds)

        out_f = open(out_fp, "w")

        for input_token, labels in zip(tokens, preds):
            token_str = input_token[1]
            labels.sort()
            str_labels = ";".join(labels)

            out_str = f"{token_str}\t{str_labels}"
            out_f.write(out_str)
        out_f.close()

    def evaluate(
        self,
        test_tokens: List[Tuple[str]],
        unimorph_labels: List[List[str]],
        prediction_fp: str,
    ):
        print("Token prediction output file: ", prediction_fp)
        X = self.embedder.transform(test_tokens)

        # remove errors
        new_X = []

        for x in X:
            if x == None:
                x = [0] * self.num_fts
            new_X.append(x)

        preds = self.clf.predict(new_X)
        preds = self.mlb.inverse_transform(preds)

        evaluation: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        out_f = open(prediction_fp, "w")
        out_f.write("TOKEN\tPREDICTION\tTRUTH")
        for token, true_labels, pred_labels in zip(test_tokens, unimorph_labels, preds):
            token = token[1]
            pred_labels = list(pred_labels)
            for tl in true_labels:
                if tl in pred_labels:
                    evaluation[tl]["tp"] += 1
                else:
                    evaluation[tl]["fn"] += 1

            for pred_l in pred_labels:
                if pred_l not in true_labels:
                    evaluation[pred_l]["fp"] += 1

            true_str = ";".join(true_labels)
            pred_str = ";".join(pred_labels)
            out_f.write(f"{token}\t{true_str}\t{pred_str}\n")
        out_f.close()

        # calculate summary statistics for each class
        print("---------------------------------")
        print("label\tprecision\trecall")
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
            print(f"{label}\t{prec}\t{rec}")


def parse_args():
    # this function only exists for testing purposes
    parser = argparse.ArgumentParser()
    parser.add_argument("src_suffix_embedding_fp")
    parser.add_argument("tgt_suffix_embedding_fp")

    parser.add_argument("src_morfessor_model_fp")
    parser.add_argument("tgt_morfessor_model_fp")

    parser.add_argument("src_unimorph_train_fp")
    parser.add_argument("src_unimorph_valid_fp")
    parser.add_argument("src_unimorph_test_fp")

    parser.add_argument("output_src_unimorph_test_fp")

    parser.add_argument("segment_method")
    parser.add_argument("--tgt_unimorph_test_fp")
    parser.add_argument("--output_tgt_unimorph_test_fp")

    args = parser.parse_args()

    src_embed = args.src_suffix_embedding_fp
    tgt_embed = args.tgt_suffix_embedding_fp

    src_morf = args.src_morfessor_model_fp
    tgt_morf = args.tgt_morfessor_model_fp
    segment_method = args.segment_method

    src_unimorph_train = args.src_unimorph_train_fp
    src_unimorph_valid = args.src_unimorph_valid_fp
    src_unimorph_test = args.src_unimorph_test_fp

    src_pred_fp = args.output_src_unimorph_test_fp

    clf = UniMorphCB(src_morf, tgt_morf, src_embed, tgt_embed, segment_method)

    x_train, y_train = clf.load_unimorph(src_unimorph_train)

    x_valid, y_valid = clf.load_unimorph(src_unimorph_valid)
    x_test, y_test = clf.load_unimorph(src_unimorph_test)

    clf.train(x_train, y_train, x_valid, y_valid)

    print("------------------------------------")
    print("SOURCE TEST SET EVALUATION:")
    clf.evaluate(x_test, y_test, src_pred_fp)

    if args.tgt_unimorph_test_fp:
        tgt_unimorph_test = args.tgt_unimorph_test_fp
        tgt_output = args.output_tgt_unimorph_test_fp

        print("------------------------------------")
        print("TARGET TEST SET EVALUATION:")
        tgt_x, tgt_y = clf.load_unimorph(tgt_unimorph_test, src_or_tgt="tgt")
        clf.evaluate(tgt_x, tgt_y, tgt_output)


if __name__ == "__main__":
    parse_args()
