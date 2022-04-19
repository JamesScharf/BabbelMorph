from collections import defaultdict
from enum import unique
import data_transformers as dt
from typing import DefaultDict, Dict, List, Set, Tuple
from catboost import CatBoostClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
import argparse
import numpy as np
from tqdm import tqdm
import morfessor_utils as mu
from sklearn.feature_extraction.text import CountVectorizer


def load_unimorph(
    fp: str, src_or_tgt="src"
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


class UniMorphCB(object):
    def __init__(
        self,
        src_morf_model_fp: str,
        tgt_morf_model_fp: str,
        src_bilingual_embed_fp: str,
        tgt_bilingual_embed_fp: str,
        segment_method: str,
        dictionary_mode=False,  # if true, just lookup tgt suffix-->src suffix and throw in src suffix features
        dictionary_fp=None,
        concat=False,
    ):
        self.concat = concat
        self.dictionary_fp = dictionary_fp
        self.dictionary_mode = dictionary_mode
        self.segment_method = segment_method
        self.embedder = dt.UniMorphEmbeddingTransformer(
            src_morf_model_fp,
            tgt_morf_model_fp,
            src_bilingual_embed_fp,
            tgt_bilingual_embed_fp,
            self.segment_method,
            dictionary_mode=self.dictionary_mode,
            dictionary_fp=self.dictionary_fp,
        )
        self.mlb = None
        self.label_vocab = None

    def process_labels(
        self, unimorph_labels: List[List[str]], train=False
    ) -> List[List[int]]:
        # convert N NOM SG --> one hot encoded

        if self.concat:
            unimorph_labels = [sorted(x) for x in unimorph_labels]
            unimorph_labels = [";".join(x) for x in unimorph_labels]

            if self.label_vocab == None:
                self.label_vocab = set()
                self.label_vocab.update(unimorph_labels)

            for i, ul in tqdm(enumerate(unimorph_labels)):
                if ul not in self.label_vocab:
                    unimorph_labels[i] = list(self.label_vocab)[0]
            return unimorph_labels

        if self.mlb == None:
            self.mlb = MultiLabelBinarizer()
            proc_labels = self.mlb.fit_transform(unimorph_labels)

        else:
            proc_labels = self.mlb.transform(unimorph_labels)

        return proc_labels

    def load_unimorph(
        self, fp: str, src_or_tgt="src"
    ) -> List[Tuple[List[Tuple[str, str]], List[str]]]:
        return load_unimorph(fp, src_or_tgt=src_or_tgt)

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

        base = CatBoostClassifier(iterations=100, boosting_type="Plain")
        if self.concat:
            print("Fitting")
            base.fit(
                new_X,
                new_y,
                # eval_set=(new_X_valid, new_y_valid),
            )
            self.clf = base
        else:
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
        # if dictionary_mode=True, then we're going to use the source embeddings for a given target word based on a lookup table

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
        if self.concat == False:
            preds = self.mlb.inverse_transform(preds)

        if self.concat:
            preds = [y.split(";") for y in preds]

        evaluation: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        out_f = open(prediction_fp, "w")
        out_f.write("TOKEN\tTRUTH\tPREDICTION")

        unique_labels = set()
        for true_labels, pred_labels in zip(unimorph_labels, preds):
            unique_labels.update(true_labels)
            unique_labels.update(pred_labels)

        num_perfect = 0
        n = 0
        for token, true_labels, pred_labels in zip(test_tokens, unimorph_labels, preds):

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
            token = token[1]
            out_f.write(f"{token}\t{true_str}\t{pred_str}\n")
        out_f.close()

        line_acc = num_perfect / n
        line_acc = round(line_acc, 3)

        # calculate summary statistics for each class
        print("label\tprecision\trecall\tfalse_pos_rate\tacc\tperfmatch")
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


class ProjectionBootstrapper(object):
    def __init__(
        self,
        src_unimorph_train_fp: str,
        tgt_unimorph_test_fp: str,
        word_dictionary_fp: str,
        prediction_output_fp: str,
    ):
        self.mlb = None
        self.feature_mapper = None
        self.dict = self.load_dictionary(word_dictionary_fp)

        src_x, src_y = load_unimorph(src_unimorph_train_fp)
        src_x = [x[1] for x in src_x]
        src_clf = self.train_classifier(src_x, src_y)
        pred_x, pred_y = self.classify_and_copy(src_clf, self.dict)
        tgt_classifier = self.train_classifier(pred_x, pred_y)
        tgt_x, tgt_y = load_unimorph(tgt_unimorph_test_fp)
        tgt_x = [x[1] for x in tgt_x]
        tgt_pred_y = self.apply_classifier(tgt_classifier, tgt_x)

        self.evaluate(tgt_x, tgt_y, tgt_pred_y, prediction_output_fp)

    def load_dictionary(self, dictionary_fp: str) -> DefaultDict[str, List[str]]:
        # note that the output is a TGT --> List[src]
        f = open(dictionary_fp)
        lns = f.readlines()
        f.close()

        out: DefaultDict[str, List[str]] = defaultdict(list)

        for ln in lns:
            src, tgt = ln.split()
            out[tgt].append(src)

        return out

    def preprocess_labels(self, unimorph_labels: List[List[str]]):
        if self.mlb == None:
            self.mlb = MultiLabelBinarizer()
            proc_labels = self.mlb.fit_transform(unimorph_labels)

        else:
            proc_labels = self.mlb.transform(unimorph_labels)

        return proc_labels

    def train_classifier(self, train_x: List[str], train_y: List[List[str]]):
        proc_y = self.preprocess_labels(train_y)

        train_x.append(train_x[0])
        train_x = self.featurize(train_x)

        # remove errors with fake entry
        proc_y = list(proc_y)
        proc_y.append(len(proc_y[0]) * [1])

        base = CatBoostClassifier(iterations=100, boosting_type="Plain")
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

        clf = ClassifierChain(base, order=order)
        clf.fit(train_x, proc_y)
        return clf

    def featurize(self, tokens: List[str]):
        # for now, just get n last suffix
        suffixized: List[Dict[str, int]] = []

        for tok in tokens:
            suf_fts = []
            for k in range(1, 5):
                suf = mu.suffix(tok, k)
                suf_fts.append(suf)
            suf_fts = " ".join(suf_fts)

            suffixized.append(suf_fts)

        if self.feature_mapper == None:
            self.feature_mapper = CountVectorizer(binary=True)
            self.feature_mapper.fit(suffixized)

        transformed = self.feature_mapper.transform(suffixized)
        return transformed

    def apply_classifier(self, classifier, x: List[str]) -> List[List[str]]:
        x = self.featurize(x)
        res = classifier.predict(x)
        return self.mlb.inverse_transform(res)

    def classify_and_copy(
        self, src_classifier, word_dictionary: Dict[str, List[str]]
    ) -> Tuple[List[str], Set[str]]:
        # bootstrap training data for target
        # dictionary is tgt-->src
        # get two lists of (w, features) for training

        y: List[Set[str]] = []

        src_words = []
        for tgt_w in word_dictionary.keys():
            src_ws = word_dictionary[tgt_w]
            src_words.extend(src_ws)

        preds = self.apply_classifier(src_classifier, src_words)

        lookup = {}

        for src_w, pred_y in zip(src_words, preds):
            lookup[src_w] = pred_y

        for tgt_w in word_dictionary.keys():
            src_ws = word_dictionary[tgt_w]
            res = [lookup[sw] for sw in src_ws]
            set_res = set()
            for r in res:
                set_res.update(r)
            y.append(list(set_res))

        return list(word_dictionary.keys()), y

    def evaluate(
        self,
        test_tokens: List[str],
        true_y: List[Set[str]],
        pred_y: List[Set[str]],
        prediction_fp: str,
    ):
        print("Token prediction output file: ", prediction_fp)

        evaluation: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        out_f = open(prediction_fp, "w")
        out_f.write("TOKEN\tTRUTH\tPREDICTION")

        preds = pred_y
        unimorph_labels = true_y
        unique_labels = set()
        for true_labels, pred_labels in zip(unimorph_labels, preds):
            unique_labels.update(true_labels)
            unique_labels.update(pred_labels)

        num_perfect = 0
        n = 0
        for token, true_labels, pred_labels in zip(test_tokens, unimorph_labels, preds):

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
            token = token[1]
            out_f.write(f"{token}\t{true_str}\t{pred_str}\n")
        out_f.close()

        line_acc = num_perfect / n
        line_acc = round(line_acc, 3)

        # calculate summary statistics for each class
        print("label\tprecision\trecall\tfalse_pos_rate\tacc\tperfmatch")
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

    parser.add_argument("--dictionary_mode", help="Either TRUE or FALSE")

    parser.add_argument("--dictionary_fp")

    parser.add_argument("--concatenate")

    parser.add_argument("--project_and_bootstrap")

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

    if args.dictionary_mode == "TRUE":
        dictionary_mode = True
        dictionary_fp = args.dictionary_fp
    else:
        dictionary_mode = False
        dictionary_fp = None

    if args.concatenate == "TRUE":
        concat = True
    else:
        concat = False

    if args.project_and_bootstrap:
        clf = ProjectionBootstrapper(
            src_unimorph_train,
            args.tgt_unimorph_test_fp,
            dictionary_fp,
            args.output_tgt_unimorph_test_fp,
        )
    else:
        clf = UniMorphCB(
            src_morf,
            tgt_morf,
            src_embed,
            tgt_embed,
            segment_method,
            dictionary_mode=dictionary_mode,
            dictionary_fp=dictionary_fp,
            concat=concat,
        )

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
