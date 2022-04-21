# Make a classifier for UniMorph features utilizing only
# the provided text embeddings
# Take a multi-output approach
from typing import Dict, List, Set, Tuple
import torch
import numpy as np
import unimorph_dataset as ud
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torchmetrics import F1Score, HammingDistance
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TextClassifier(pl.LightningModule):
    def __init__(
        self,
        src_iso: str,
        tgt_iso: str,
        embed_dim,
        use_embeds=False,
        validate_on_generated_tgt=False,
    ):

        # if validate_on_generated_tgt=True, then our validation script will use TARGET data
        # in the ./data/unimorph/generated/{tgt_iso}
        super(TextClassifier, self).__init__()

        self.validate_generated = validate_on_generated_tgt
        self.lr = 3e-4
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.use_embeds = use_embeds

        if use_embeds:
            vocab_size = 400
        else:
            vocab_size = self.get_vocab_size(self.src_iso)
        num_outputs = self.get_num_outputs(self.src_iso)

        print(self.src_iso)
        print(self.tgt_iso)
        print(vocab_size)
        print(num_outputs)
        print(embed_dim)

        # extract a bunch of stuff from the vectorization
        # dataset
        # There's really no c/on way to do this...
        self.extract_data_from_dataset()
        # (many side effects)

        # setup layers
        if self.use_embeds == False:
            self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim)
            self.lstm = torch.nn.LSTM(
                embed_dim, embed_dim, num_layers=1, bidirectional=True
            )
        else:
            self.lstm = torch.nn.LSTM(
                vocab_size, embed_dim, num_layers=1, bidirectional=True
            )
        self.linear = torch.nn.Linear(2 * embed_dim, num_outputs)
        self.softmax = torch.nn.LogSoftmax(dim=1)

        # metrics
        self.f1 = F1Score(num_classes=len(self.one_label2idx), multiclass=False)
        self.hamming = HammingDistance()

        self.src_test_mode = True

    def extract_data_from_dataset(self):
        # obtain label mapper
        print("Obtaining label mapper")
        dataset = ud.UniMorphDataset(self.src_iso, self.tgt_iso)
        # extract the vectorization method for potential deployment
        # down the road
        self.vect_method = dataset.vect_method
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

    def init_weights(self):
        initrange = 0.5
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()

    def forward(self, x):

        if self.use_embeds == False:
            x = self.embedding(x)
        # dropped_embeds = self.dropout(embedded)
        lstm_out, _ = self.lstm(x)
        l2_out = self.linear(lstm_out)
        preds = self.softmax(l2_out)
        return preds

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

        out = torch.zeros(batch_size, num_labels, dtype=torch.int32)
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
        if self.use_embeds:
            token_tensor = self.vect_method.embed_many_and_merge(tokens, src_or_tgt)
        else:
            token_tensor = self.vect_method.tokens2tensor(tokens)

        return token_tensor

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, log_name="test")

    def make_dataloader(
        self, src_iso: str, tgt_iso: str, use_embeddings: bool, mode: str
    ):
        # mode must be one of "train", "valid", "test"
        print(mode)
        dataset = ud.UniMorphDataset(
            src_iso, tgt_iso, use_embeddings=use_embeddings, mode=mode
        )

        if mode == "train":
            shuffle = True
        else:
            shuffle = False

        loader = ud.DataLoader(dataset, batch_size=64, shuffle=shuffle, num_workers=10)

        return loader

    def train_dataloader(self):
        return self.make_dataloader(
            self.src_iso, self.tgt_iso, self.use_embeds, "train_src"
        )

    def val_dataloader(self):

        if self.validate_generated:
            return self.make_dataloader(
                self.src_iso, self.tgt_iso, self.use_embeds, "generated_tgt"
            )
        else:
            return self.make_dataloader(
                self.src_iso, self.tgt_iso, self.use_embeds, "valid_src"
            )

    def test_dataloader(self):

        if self.src_test_mode:
            mode = "test_src"
        else:
            mode = "test_tgt"
        return self.make_dataloader(self.src_iso, self.tgt_iso, self.use_embeds, mode)

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

        if self.use_embeds:
            src_or_tgt = xs[0][0]
            tensorized_x = self.vect_method.embed_many_and_merge(
                [x[1] for x in xs], src_or_tgt
            )
        else:
            tensorized_x = self.vect_method.tokens2tensor([x[1] for x in xs])

        y_hat = self(tensorized_x)
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
    src_iso: str, tgt_iso: str, use_embeddings: bool, validate_on_generated_tgt=False
):
    # if validate_on_generated_tgt=True, then use the data in unimorph/generated/tgt for
    # validation step

    trainer = pl.Trainer(
        max_epochs=15,
        gpus=1,
        progress_bar_refresh_rate=20,
        callbacks=[EarlyStopping(monitor="val_hamming", mode="min", verbose=True)],
        auto_lr_find=True,
    )
    model = TextClassifier(
        src_iso,
        tgt_iso,
        128,
        use_embeds=use_embeddings,
        validate_on_generated_tgt=validate_on_generated_tgt,
    )
    trainer.fit(model)
    return trainer, model


def evaluate_classifier(trainer, model, src_or_tgt: str):
    if src_or_tgt == "src":
        src_test = trainer.test(model, verbose=True)
        return src_test
    else:
        model.enable_tgt_test_mode()
        tgt_test = trainer.test(model, verbose=True)
        return tgt_test


trainer, model = make_classifier(
    "lat", "spa", use_embeddings=False, validate_on_generated_tgt=False
)

print(evaluate_classifier(trainer, model, "src"))
print(evaluate_classifier(trainer, model, "tgt"))

xs = [
    ("src", "canis"),
    (
        "src",
        "canibus",
    ),
    ("src", "cane"),
]