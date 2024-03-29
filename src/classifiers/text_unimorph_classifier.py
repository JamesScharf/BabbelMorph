# Make a classifier for UniMorph features utilizing only
# the provided text embeddings
# Take a multi-output approach
from genericpath import exists
from json import load
from multiprocessing import context
from typing import Dict, List, Set, Tuple
import torch
import numpy as np
import unimorph_dataset as ud
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import F1Score, HammingDistance
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import ConcatDataset


class TextClassifier(pl.LightningModule):
    def __init__(
        self,
        src_iso: str,
        tgt_iso: str,
        embed_dim,
        use_embeds=False,
        validate_on_generated_tgt=False,
        train_on_generated=False,
        annotation_source=None,
    ):

        # if validate_on_generated_tgt=True, then our validation script will use TARGET data
        # in the ./data/unimorph/generated/{tgt_iso}
        super(TextClassifier, self).__init__()

        self.validate_generated = validate_on_generated_tgt
        self.annotation_source = annotation_source
        self.train_generated = train_on_generated
        # self.lr = 3e-4
        self.lr = 0.001
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.use_embeds = use_embeds

        self.vocab_size = self.get_vocab_size(self.src_iso)
        self.num_outputs = self.get_num_outputs(self.src_iso)

        self.batch_size = 64
        self.embed_dim = embed_dim

        # extract a bunch of stuff from the vectorization
        # dataset
        # There's really no c/on way to do this...
        self.extract_data_from_dataset()
        # (many side effects)

        self.hidden_size = 128
        # setup layers for token
        self.token_embedding = torch.nn.Embedding(
            self.vocab_size, embed_dim, padding_idx=0
        )
        self.token_lstm = torch.nn.LSTM(
            embed_dim,
            self.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.token_linear = torch.nn.Sequential(
            torch.nn.Linear(2 * self.hidden_size, 2 * self.num_outputs), torch.nn.ReLU()
        )

        # set up layers for embeddings
        # going to use simple feedforward for this

        self.embedding_reduction = torch.nn.Sequential(
            torch.nn.Linear(400, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, self.num_outputs * 2),
            torch.nn.ReLU(),
        )

        # now we need a layer to combine the "votes"
        self.linear_vote = torch.nn.Linear(4 * self.num_outputs, self.num_outputs)

        # metrics
        self.f1 = F1Score(num_classes=len(self.one_label2idx), multiclass=False)
        self.hamming = HammingDistance()

        self.src_test_mode = True

    def extract_data_from_dataset(self):
        # obtain label mapper
        dataset = ud.UniMorphDataset(self.src_iso, self.tgt_iso)
        # extract the vectorization method for potential deployment
        # down the road
        self.vect_method = dataset.vect_method
        self.morph = dataset.morph
        # extract label2idx
        label2idx = dataset.label2idx
        self.idx2label = self.make_idx_to_label(label2idx)

        # build one label to idx converter
        # e.g., "SG" --> NONE
        self.one_label2idx = {}
        self.one_label2idx["UNK"] = 0
        count = 1

        for _, label in self.idx2label.items():
            labels = label.split("+")
            for ft_label in labels:
                if ft_label not in self.one_label2idx.keys():
                    self.one_label2idx[ft_label] = count
                    count += 1

        dataset = None

    def enable_tgt_test_mode(self):
        # flip src_test_mode to False
        # meaning that we test on tgt language
        self.src_test_mode = False

    def make_idx_to_label(self, label2idx):
        idx2label = {}
        for k, v in label2idx.items():
            idx2label[v] = k
        return idx2label

    # def init_weights(self):
    #    initrange = 0.5
    #    self.linear.weight.data.uniform_(-initrange, initrange)
    #    self.linear.bias.data.zero_()

    def forward(self, x):

        # handle token stuff first
        # Process: Embedding --> LSTM --> linear layer with predictions
        tok_x = self.token_embedding(x[0])
        tok_x = pack_padded_sequence(
            tok_x,
            torch.tensor([12] * len(tok_x)),
            batch_first=True,
            enforce_sorted=True,
        )
        lstm_output, (hidden, cell) = self.token_lstm(tok_x)
        tok_hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        tok_lin_out = self.token_linear(tok_hidden)

        # now handle contextual embeddings
        context_embeds = x[1]
        embed_out = self.embedding_reduction(context_embeds)

        # now combine the votes by concatenating them with a
        # linear layer at the end
        combo_votes = torch.cat((tok_lin_out, embed_out), dim=1)
        return self.linear_vote(combo_votes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def convert_tensor_to_multilabel(self, batch) -> torch.Tensor:
        # convert a given target tensor to a multi-label problem
        batch_lst = batch.tolist()

        batch_size = batch.size()[0]
        num_labels = len(self.one_label2idx.keys())

        out = torch.zeros(batch_size, num_labels, dtype=torch.int64)
        for row_num, grouped_label in enumerate(batch_lst):
            grouped_label_str = self.idx2label[grouped_label]
            splt_label = grouped_label_str.split("+")

            for label in splt_label:
                idx = self.one_label2idx[label]
                out[row_num, idx] = 1
        return out

    def validation_step(self, batch, batch_idx, log_name="val"):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log(log_name + "_loss", loss)
        preds = torch.argmax(y_hat, dim=1)

        # get multi-label metrics now
        multilabel_preds = self.convert_tensor_to_multilabel(preds)
        multilabel_y = self.convert_tensor_to_multilabel(y)
        self.f1(multilabel_preds, multilabel_y)
        self.hamming(multilabel_preds, multilabel_y)

        # self.precision(preds, y)
        # self.recall(preds, y)
        # self.f1(preds, y)
        # self.accuracy(preds, y)

        # self.log(log_name + "_precision", self.precision)
        # self.log(log_name + "_recall", #self.recall)
        self.log(log_name + "_f1", self.f1)
        self.log(log_name + "_hamming", self.hamming)
        # self.log(log_name + "_acc", self.accuracy)

        return loss

    def process_tokens(self, tokens: List[str], src_or_tgt: str) -> torch.Tensor:
        # convert tokens to features
        embeds = self.morph.embed_many_and_merge(tokens, src_or_tgt)
        token_tensor = self.vect_method.tokens2tensor(tokens)

        return (embeds, token_tensor)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, log_name="test")

    def make_dataloader(
        self, src_iso: str, tgt_iso: str, mode: str, use_generated_tgt=False
    ):
        # mode must be one of "train", "valid", "test"
        dataset = ud.UniMorphDataset(src_iso, tgt_iso, mode=mode)

        if mode == "train_src" and use_generated_tgt:

            # then add generated data
            gen_dataset = ud.UniMorphDataset(
                self.src_iso,
                self.tgt_iso,
                mode="generated_tgt",
                annotation_source=self.annotation_source,
            )
            dataset = ConcatDataset([dataset, gen_dataset])

        if mode == "train_src":
            shuffle = True
        else:
            shuffle = False

        loader = ud.DataLoader(dataset, batch_size=64, shuffle=shuffle, num_workers=10)

        return loader

    def train_dataloader(self):
        train_src = self.make_dataloader(
            self.src_iso,
            self.tgt_iso,
            "train_src",
            use_generated_tgt=self.train_generated,
        )
        return train_src

    def val_dataloader(self):

        if self.validate_generated:
            return self.make_dataloader(self.src_iso, self.tgt_iso, "generated_tgt")
        else:
            return self.make_dataloader(self.src_iso, self.tgt_iso, "valid_src")

    def test_dataloader(self):

        if self.src_test_mode:
            mode = "test_src"
        else:
            mode = "test_tgt"
        return self.make_dataloader(self.src_iso, self.tgt_iso, mode)

    def get_vocab_size(self, iso: str) -> int:
        fp = f"./data/unimorph/train/{iso}"
        f = open(fp, "r")

        unique_chars = set()
        for ln in f:
            splt_ln = ln.split()
            token = splt_ln[1]
            unique_chars.update(set(token))

        return len(unique_chars)

    def get_num_outputs(self, iso: str) -> int:
        fp = f"./data/unimorph/train/{iso}"
        f = open(fp, "r")

        unique_labels = set()
        for ln in f:
            splt_ln = ln.split()
            labels = splt_ln[2:]
            labels.sort()
            str_lab = "+".join(labels)
            unique_labels.add(str_lab)
        return len(unique_labels)

    def predict(self, xs: List[Tuple[str, str]]):
        # convert strings to encodings
        # xs are expected to be in format [(src, TOKEN), (src, token)] or [(tgt, token), (tgt, token)]

        src_or_tgt = xs[0][0]
        morph_x = torch.tensor(
            self.morph.embed_many_and_merge([x[1] for x in xs], src_or_tgt)
        )
        tok_x = torch.tensor(self.vect_method.tokens2tensor([x[1] for x in xs]))

        y_hat = self((tok_x, morph_x))
        label_nums = torch.argmax(y_hat, dim=1)
        # convert those labels to predictions
        labeled_preds: List[Set[str]] = []
        label_num_lst = label_nums.tolist()

        for ln in label_num_lst:
            raw_label = self.idx2label[ln]
            splt_label = raw_label.split("+")
            labeled_preds.append(set(splt_label))

        return labeled_preds


def make_classifier(
    src_iso: str, tgt_iso: str, train_on_generated=False, annotation_source=None
):
    # if validate_on_generated_tgt=True, then use the data in unimorph/generated/tgt for
    # validation step

    if train_on_generated and annotation_source != None:
        model_fp = f"./data/trained_classifiers/{src_iso}_{tgt_iso}_bootstrapped_{annotation_source}"
    else:
        model_fp = f"./data/trained_classifiers/{src_iso}_{tgt_iso}"
    print(model_fp)

    if exists(model_fp):
        model = TextClassifier.load_from_checkpoint(model_fp)
        model.eval()
        trainer = None
    else:
        trainer = pl.Trainer(
            max_epochs=60,
            gpus=1,
            progress_bar_refresh_rate=20,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True)],
            auto_lr_find=True,
            precision=16,
            default_root_dir=model_fp,
        )
        model = TextClassifier(
            src_iso,
            tgt_iso,
            128,
            validate_on_generated_tgt=False,
            train_on_generated=train_on_generated,
            annotation_source=annotation_source,
        )
        trainer.tune(model)
        trainer.fit(model)
        trainer.save_checkpoint(
            f"./data/trained_classifiers/{src_iso}_{tgt_iso}/final.ckpt"
        )
    return trainer, model


def evaluate_classifier(
    model_checkpt_path: str,
    src_iso: str,
    tgt_iso: str,
    src_or_tgt: str,
):
    model = TextClassifier.load_from_checkpoint(
        model_checkpt_path,
        src_iso=src_iso,
        tgt_iso=tgt_iso,
        embed_dim=128,
    )
    model.eval()

    trainer = pl.Trainer(gpus=1)
    if src_or_tgt == "src":
        print("Testing on src")
        model.mode = "test_src"
        src_test = trainer.test(model)
        return src_test[0]
    else:
        print("Testing on tgt")
        model.enable_tgt_test_mode()
        tgt_test = trainer.test(model)
        return tgt_test[0]


if __name__ == "__main__":
    print("src")
    res = evaluate_classifier(
        "./data/trained_classifiers/nno_swe/final.ckpt",
        "nno",
        "swe",
        False,
        "src",
    )
    print(res)
    print("tgt")
    res = evaluate_classifier(
        "./data/trained_classifiers/nno_swe/final.ckpt",
        "nno",
        "swe",
        False,
        "tgt",
    )
    print(res)
