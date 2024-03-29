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
        to_ascii=True,
        mode="train_src",
        annotation_source=None,  # only relevant for generated
    ):
        # use_embeddings=True: uses MorphEmbeddingLoader instead of char2vec
        # use_embeddings=False: use char 2 vec
        self.annotation_source = annotation_source
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.morph = MorphEmbeddingLoader(src_iso, tgt_iso)
        self.vect_method = TokenLoader(self.src_iso, to_ascii=to_ascii)

        self.mode = mode

        self.determine_active_dataset(src_iso, tgt_iso)

    def determine_active_dataset(self, src_iso, tgt_iso):

        train_dataset = self.load_raw_data(self.src_iso, "train")
        self.label2idx = self.make_label_index(train_dataset)
        if self.mode == "train_src":
            self.active_dataset = self.process_data(train_dataset, src_iso)
        elif self.mode == "train_tgt":
            train_dataset = self.load_raw_data(self.tgt_iso, "valid")
            self.active_dataset = self.process_data(train_dataset, tgt_iso)
        elif self.mode == "valid_src":
            valid_dataset = self.load_raw_data(self.src_iso, "valid")
            self.active_dataset = self.process_data(valid_dataset, src_iso)
        elif self.mode == "valid_tgt":
            valid_dataset = self.load_raw_data(self.tgt_iso, "valid")
            self.active_dataset = self.process_data(valid_dataset, tgt_iso)
        elif self.mode == "test_src":
            test_dataset = self.load_raw_data(self.src_iso, "test")
            self.active_dataset = self.process_data(test_dataset, src_iso)
        elif self.mode == "test_tgt":
            test_dataset = self.load_raw_data(self.tgt_iso, "test")
            self.active_dataset = self.process_data(test_dataset, tgt_iso)
        elif self.mode == "generated_tgt":
            # returns the data that was auto-generated by the projection
            # process
            generated_dataset = self.load_raw_data(self.tgt_iso, "generated")
            self.active_dataset = self.process_data(generated_dataset, tgt_iso)

        else:
            raise ArgumentError("mode must be one of train, valid, test")

    def load_raw_data(self, iso: str, mode: str) -> List[Tuple[str, List[str]]]:
        # if in train mode, load the train set of the src iso
        # if in test mode, load the test set of the tgt iso
        if mode == "generated":
            if self.annotation_source == "src":
                fp = f"./data/unimorph/{mode}/{self.src_iso}_{self.tgt_iso}.src_labels"
            else:
                fp = f"./data/unimorph/{mode}/{self.src_iso}_{self.tgt_iso}.tgt_labels"
        else:
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

        src_or_tgt = "src" if iso == self.src_iso else "tgt"
        morph_tensor = self.morph.embed_many_and_merge(tokens, src_or_tgt)
        token_tensor = self.vect_method.tokens2tensor(tokens)

        return (token_tensor, morph_tensor), conv_labels_tensor

    def __len__(self):
        return self.active_dataset[0][0].size()[0]

    def __getitem__(self, idx):

        x = (self.active_dataset[0][0][idx], self.active_dataset[0][1][idx])
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
        return self.make_dataloader(self.src_iso, self.tgt_isos, "train")

    def val_dataloder(self):
        return self.make_dataloader(self.src_iso, self.tgt_isos, "valid")

    def test_dataloader(self):
        return self.make_dataloader(self.src_iso, self.tgt_iso, "test")

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
