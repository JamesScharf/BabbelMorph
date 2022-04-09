from collections import defaultdict
import enum
import data_transformers as dt
from typing import List, Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
import argparse
from sklearn.metrics import classification_report


class UniMorphCB(object):
    def __init__(
        self,
        src_morf_model_fp: str,
        tgt_morf_model_fp: str,
        src_bilingual_embed_fp: str,
        tgt_bilingual_embed_fp: str,
    ):
        self.embedder = dt.UniMorphEmbeddingTransformer(
            src_morf_model_fp,
            tgt_morf_model_fp,
            src_bilingual_embed_fp,
            tgt_bilingual_embed_fp,
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

        for x, y in zip(X, proc_labels):
            if x != None:
                new_X.append(x)
                new_y.append(y)

        # remove errors
        new_X_valid = []
        new_y_valid = []

        for x, y in zip(valid_X, valid_labels):
            if x != None:
                new_X_valid.append(x)
                new_y_valid.append(y)

        base = CatBoostClassifier(iterations=200)  # , task_type="GPU", devices="0:1"

        # order doesn't matter except we want POS first
        pos = ["N", "ADJ", "V"]

        order = []
        for p in pos:
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

    def evaluate(self, test_tokens: List[Tuple[str]], unimorph_labels: List[List[str]]):

        true_proc_labels = self.process_labels(unimorph_labels)
        X = self.embedder.transform(test_tokens)

        # remove errors
        new_X = []
        new_y = []

        for x, y in zip(X, true_proc_labels):
            if x != None:
                new_X.append(x)
                new_y.append(y)

        true_proc_labels = new_y

        preds = self.clf.predict(new_X).tolist()

        print("Beginning evaluation...")
        for i, cl in enumerate(list(self.mlb.classes_)):
            pred_col = [x[i] for x in preds]
            true_col = [x[i] for x in true_proc_labels]
            print(cl)
            print(classification_report(true_col, pred_col, zero_division=0))

        # now print out the extracted tokens
        print("PREDICTED TOKENS")
        print("TOKEN\tPRED_LABELS\tTRUE_LABELS")
        for token, pred_labels, true_labels in zip(test_tokens, preds, unimorph_labels):
            extracted_pred = ";".join(self.mlb.inverse_transform(pred_labels))
            print(f"{token}\t{extracted_pred}\t{true_labels}")


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

    parser.add_argument("--tgt_unimorph_test_fp")

    args = parser.parse_args()

    src_embed = args.src_suffix_embedding_fp
    tgt_embed = args.tgt_suffix_embedding_fp

    src_morf = args.src_morfessor_model_fp
    tgt_morf = args.tgt_morfessor_model_fp

    src_unimorph_train = args.src_unimorph_train_fp
    src_unimorph_valid = args.src_unimorph_valid_fp
    src_unimorph_test = args.src_unimorph_test_fp

    clf = UniMorphCB(src_morf, tgt_morf, src_embed, tgt_embed)

    x_train, y_train = clf.load_unimorph(src_unimorph_train)

    x_valid, y_valid = clf.load_unimorph(src_unimorph_valid)
    x_test, y_test = clf.load_unimorph(src_unimorph_test)

    clf.train(x_train, y_train, x_valid, y_valid)

    print("------------------------------------")
    print("SOURCE TRAIN SET EVALUATION:")
    clf.evaluate(x_train, y_train)

    print("------------------------------------")
    print("SOURCE TEST SET EVALUATION:")
    clf.evaluate(x_test, y_test)

    if args.tgt_unimorph_test_fp:
        tgt_unimorph_test = args.tgt_unimorph_test_fp

        print("------------------------------------")
        print("TARGET TEST SET EVALUATION:")
        tgt_x, tgt_y = clf.load_unimorph(tgt_unimorph_test, src_or_tgt="tgt")
        clf.evaluate(tgt_x, tgt_y)


if __name__ == "__main__":
    parse_args()
