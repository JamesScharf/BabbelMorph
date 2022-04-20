# Make a classifier for UniMorph features utilizing only
# the provided text embeddings
# Take a multi-output approach
import torch
import numpy as np
import unimorph_dataset as ud
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch.nn.functional as F
from torchmetrics import Precision, Recall, F1Score, Accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class TextClassifier(pl.LightningModule):
    def __init__(self, src_iso: str, tgt_iso: str, embed_dim, use_embeds=False):
        super(TextClassifier, self).__init__()
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.use_embeds = use_embeds
        vocab_size = self.get_vocab_size(self.src_iso)
        num_outputs = self.get_num_outputs(self.src_iso)

        print(self.src_iso)
        print(self.tgt_iso)
        print(vocab_size)
        print(num_outputs)
        print(embed_dim)
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim)
        self.l1 = torch.nn.Linear(embed_dim, embed_dim)
        self.l2 = torch.nn.Linear(embed_dim, num_outputs)
        # metrics
        # self.precision = Precision(num_classes=num_outputs)
        # self.recall = Recall(num_classes=num_outputs)
        self.f1 = F1Score(num_classes=num_outputs)
        # self.accuracy = Accuracy(num_classes=num_outputs)

    def init_weights(self):
        initrange = 0.5
        self.l1.weight.data.uniform_(-initrange, initrange)
        self.l1.bias.data.zero_()
        self.l2.weight.data.uniform_(-initrange, initrange)
        self.l2.bias.data.zero_()

    def forward(self, text_as_tensor):
        embedded = self.embedding(text_as_tensor)
        l1_out = self.l1(embedded)
        l2_out = self.l2(l1_out)
        preds = torch.relu(l2_out)
        return preds

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx, log_name="val"):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log(log_name + "_loss", loss)
        preds = torch.argmax(y_hat, dim=1)

        # self.precision(preds, y)
        # self.recall(preds, y)
        self.f1(preds, y)
        # self.accuracy(preds, y)

        # self.log(log_name + "_precision", self.precision)
        # self.log(log_name + "_recall", #self.recall)
        self.log(log_name + "_f1", self.f1)
        # self.log(log_name + "_acc", self.accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, log_name="test")

    def make_dataloader(
        self, src_iso: str, tgt_iso: str, use_embeddings: bool, mode: str
    ):
        # mode must be one of "train", "valid", "test"
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
            self.src_iso, self.tgt_iso, self.use_embeds, "train"
        )

    def val_dataloader(self):
        return self.make_dataloader(
            self.src_iso, self.tgt_iso, self.use_embeds, "valid"
        )

    def test_dataloader(self):
        return self.make_dataloader(self.src_iso, self.tgt_iso, self.use_embeds, "test")

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


src_iso = "rus"
tgt_iso = "ukr"
trainer = pl.Trainer(
    max_epochs=10,
    gpus=1,
    progress_bar_refresh_rate=20,
    callbacks=[EarlyStopping(monitor="val_loss", mode="min", verbose=True)],
)
model = TextClassifier(src_iso, tgt_iso, 64)
trainer.fit(model)
trainer.test(model, verbose=True)
