from argparse import ArgumentError
from cProfile import label
from typing import List, Tuple, Set, Dict
from torch.utils import data
from embedding_loader import MorphEmbeddingLoader
from token_loader import TokenLoader
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


class UniMorphDataset(data.Dataset):
    def __init__(
        self,
        src_iso: str,
        tgt_iso: str,
        use_embeddings=False,
        to_ascii=True,
        mode="train",  # one of "train", "valid", "test"
    ):
        # use_embeddings=True: uses MorphEmbeddingLoader instead of char2vec
        # use_embeddings=False: use char 2 vec
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.use_embeds = use_embeddings
        if use_embeddings:
            self.vect_method = MorphEmbeddingLoader(src_iso, tgt_iso)
        else:
            self.vect_method = TokenLoader(self.src_iso, to_ascii=to_ascii)

        self.mode = mode

        train_dataset = self.load_raw_data(self.src_iso, "train")
        self.label2idx = self.make_label_index(train_dataset)
        if self.mode == "train":
            self.active_dataset = self.process_data(train_dataset, src_iso)
        elif self.mode == "valid":
            valid_dataset = self.load_raw_data(self.src_iso, "valid")
            self.active_dataset = self.process_data(valid_dataset, src_iso)
        elif self.mode == "test":
            test_dataset = self.load_raw_data(self.tgt_iso, "test")
            self.active_dataset = self.process_data(test_dataset, tgt_iso)
        else:
            raise ArgumentError("mode must be one of train, valid, test")

    def load_raw_data(self, iso: str, mode: str) -> List[Tuple[str, List[str]]]:
        # if in train mode, load the train set of the src iso
        # if in test mode, load the test set of the tgt iso
        fp = f"./data/unimorph/{mode}/{iso}"

        f = open(fp, "r")

        out: List[Tuple[str, List[str]]] = []
        for ln in f:
            splt_ln = ln.split()
            token = splt_ln[1]
            labels = splt_ln[2:]

            out.append((token, labels))

        return out

    def make_label_index(self, raw_data: List[Tuple[str, List[str]]]):
        # make label indexer
        found_labels: Set[str] = set()
        for _, labels in raw_data:

            # concatenate labels

            labels = [x for x in labels if x != "none"]
            labels.sort()
            labels = "+".join(labels)
            found_labels.add(labels)

        mapper: Dict[str, int] = {}

        label_lst = list(found_labels)
        for i, l in enumerate(label_lst):
            if i == 0:
                mapper["none"] = i
            else:
                mapper[l] = i

        return mapper

    def process_data(
        self, raw_data: List[Tuple[str, List[str]]], iso: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # convert data to the format we care about

        # seperate tokens and labels
        tokens: List[str] = []
        labels: List[List[str]] = []

        for t, ls in raw_data:
            tokens.append(t)

            ls = [x for x in ls if x != "none"]
            ls.sort()
            ls = "+".join(ls)
            labels.append(ls)

        # convert labels to the appropriate format
        num_labels: List[int] = []
        for ls in labels:
            num_label = self.label2idx.get(ls, 0)
            num_labels.append(num_label)

        conv_labels_tensor = torch.tensor(num_labels)

        # now convert tokens by whatever x vectorization method we're using
        src_or_tgt = "src" if iso == self.src_iso else "tgt"
        if self.use_embeds:
            token_tensor = self.vect_method.embed_many_and_merge(tokens, src_or_tgt)
        else:
            token_tensor = self.vect_method.tokens2tensor(tokens)

        return token_tensor, conv_labels_tensor

    def __len__(self):
        return self.active_dataset[0].size()[0]

    def __getitem__(self, idx):

        x = self.active_dataset[0][idx]
        y = self.active_dataset[1][idx]
        return x, y


class DataModule(pl.LightningDataModule):
    def __init__(self, src_iso: str, tgt_iso: str, use_embeddings: bool):
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.use_embeds = use_embeddings

    def make_dataloader(
        self, src_iso: str, tgt_iso: str, use_embeddings: bool, mode: str
    ):
        # mode must be one of "train", "valid", "test"
        dataset = UniMorphDataset(
            src_iso, tgt_iso, use_embeddings=use_embeddings, mode=mode
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=10)

        return loader

    def train_dataloader(self):
        return self.make_dataloader(
            self.src_iso, self.tgt_iso, self.use_embeds, "train"
        )

    def val_dataloder(self):
        return self.make_dataloader(
            self.src_iso, self.tgt_iso, self.use_embeds, "valid"
        )

    def test_dataloader(self):
        return self.make_dataloader(self.src_iso, self.tgt_iso, self.use_embeds, "test")

    def predict_dataloader(self):
        pass


def get_vocab_size(iso: str) -> int:
    fp = f"./data/unimorph/train/{iso}"
    f = open(fp, "r")

    unique_chars = set()
    for ln in f:
        splt_ln = ln.split()
        token = splt_ln[1]
        unique_chars.update(set(token))

    return len(unique_chars)


def get_num_outputs(iso: str) -> int:
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
