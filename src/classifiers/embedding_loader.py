# A collection of utilities for data loading
from typing import Dict, List, Tuple
from glob import glob
import torch
from tqdm import tqdm
import sys

sys.path.insert(1, "./src/utils")
import morfessor_utils as mu
import morfessor


class MorphEmbeddingLoader(object):
    # utility class for loading morphological embeddings for some language

    def __init__(
        self,
        src_iso: str,
        tgt_iso: str,
    ):
        self.src_iso = src_iso
        self.tgt_iso = tgt_iso
        self.method2fp, self.logbook = self.find_embeddings()
        self.active_embeddings = None  # the current embeddings that loaded

        self.src_morf_model = self.load_morfessor_model(src_iso)
        self.tgt_morf_model = self.load_morfessor_model(tgt_iso)

    def load_morfessor_model(self, iso):
        fp = f"./data/morfessor_models/{iso}"
        io = morfessor.MorfessorIO()
        model = io.read_any_model(fp)
        return model

    def find_embeddings(
        self,
    ) -> Tuple[Dict[Tuple[str, str], str], Dict[Tuple[str, str], bool]]:
        # obtain a method-->fp dictionary for lazy loading
        # also return a logbook which indicates whether embeddings were called last

        method2fp: Dict[str, str] = {}
        callable_logbook: Dict[str, bool] = {}

        for fp in glob(
            f"./data/crosslingual_embeddings/{self.src_iso}_{self.tgt_iso}_*"
        ):
            src_fp = f"{fp}/{self.src_iso}.vec"
            tgt_fp = f"{fp}/{self.tgt_iso}.vec"

            folder = fp.split("/")[-1]
            splt_ln = folder.split("_")
            if len(splt_ln) == 4:
                method = "_".join(splt_ln[-2:])
            elif len(splt_ln) == 3:
                method = splt_ln[-1]
            else:
                print("Skipped: ", fp)
                continue

            method2fp[("src", method)] = src_fp
            method2fp[("tgt", method)] = tgt_fp

            callable_logbook[("src", method)] = False

        return method2fp, callable_logbook

    def load_embedding(self, fp: str) -> Dict[str, torch.Tensor]:
        f = open(fp, "r")
        embed_map: Dict[str, torch.Tensor] = {}
        for ln in tqdm(f, desc="Loading embedding"):
            splt_ln = ln.split()
            w = splt_ln[0]
            vect = splt_ln[1:]
            t = torch.tensor([float(ft) for ft in vect])
            embed_map[w] = t

        f.close()

        return embed_map

    def embed(self, w: str, src_or_tgt: str, method: str) -> torch.Tensor:
        # get embeddings of word (not segmented)
        # w = token, src_or_tgt="src" or "tgt, method=some UniMorph method"

        # if not previously the last, need to load
        if not self.logbook[(src_or_tgt, method)]:
            self.active_embeddings = None
            # make all other values False
            for k, _ in self.logbook.items():
                self.logbook[k] = False

            self.logbook[(src_or_tgt, method)] = True
            method_fp = self.method2fp[(src_or_tgt, method)]
            self.active_embeddings = self.load_embedding(method_fp)

        if src_or_tgt == "src":
            segment = mu.segment_token(self.src_morf_model, w, method)
        elif src_or_tgt == "tgt":
            segment = mu.segment_token(self.tgt_morf_model, w, method)

        embeddings: List[torch.Tensor] = []
        for morph in segment.split():
            embeds = self.active_embeddings.get(morph, torch.zeros(1, 50))
            embeddings.append(embeds)

        centroid = torch.vstack(embeddings).mean(dim=0)

        return centroid

    def embed_many(
        self, words: List[str], src_or_tgt: str, method: str
    ) -> torch.Tensor:
        # run embed on many words; uses ONE method though
        # return a tensor of dimension num_words x embed_len

        embeddings: List[torch.Tensor] = []
        for w in words:
            embedding = self.embed(w, src_or_tgt, method)
            embeddings.append(embedding)

        return torch.stack(embeddings)

    def embed_many_and_merge(
        self, words: List[str], src_or_tgt: str, methods=None
    ) -> torch.Tensor:
        # get embeddings for each word and horizontally concat
        # if methods=None, assume that we're using all embedding methods

        if methods == None:
            all_methods = list(self.method2fp.keys())
            methods = [m[1] for m in all_methods if m[0] == src_or_tgt][0:2]
            # make sure method order is always the same
            methods.sort()

        method_embeds = [self.embed_many(words, src_or_tgt, m) for m in methods]
        t = torch.hstack(method_embeds)
        return t
