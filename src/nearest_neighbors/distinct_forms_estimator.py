# Obtain an estimate of the ratio of [POS] to [POS] in language a to language b
from email.policy import default
from genericpath import exists
from glob import glob
from tabnanny import verbose
from typing import List, Tuple, Dict, DefaultDict
from statistics import median
from collections import defaultdict
import editdistance as ed
from sklearn.cluster import dbscan
import numpy as np
import lang2vec.lang2vec as l2v
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm


def parse_unimorph_ln(line: str) -> List[str]:
    # format: LEMMA TOKEN LABEL1 LABEL2 LABEL3 LABEL4...
    splt_ln = line.split()
    lemma = splt_ln[0]
    token = splt_ln[1]

    if len(splt_ln) == 2:
        labels = ["ERROR"]
    else:
        labels = splt_ln[2:]

    if "N" in labels:
        pos = "N"
    elif "V" in labels:
        pos = "V"
    elif "ADJ" in labels:
        pos = "ADJ"
    else:
        pos = None

    labels.sort()

    return (pos, lemma, token, labels)


def editdistance_clustering(lemmas: List[str], iso: str, back_w=1.5) -> Dict[str, int]:
    # cluster tokens by their suffix to determine word class (e.g., gender)
    # give the last morpheme more weight since, at least in Latin, there
    # tend to be more regularity there
    # Inspired by:
    # https://aclanthology.org/2021.sigmorphon-1.12.pdf

    suffixes = [lemma.split("__")[1:] for lemma in lemmas]

    out: Dict[str, int] = {}

    # if clustering already exists, just read from there
    file_cache = f"./data/unimorph/lemma_clusters/{iso}"
    if exists(file_cache):
        clust_f = open(file_cache, "r")
        lns = clust_f.readlines()

        for ln in lns:
            lemma, clust = ln.split()
            out[lemma] = int(clust)
        return out

    def dual_edit_metric(x, y, back_w=back_w):

        # if clustering already exists, just read from there
        # perhaps back_w should be tuned...?
        # get average of distance between both parts of suffixes [0] and [1]
        i, j = int(x[0]), int(y[0])  # extract indices

        a = suffixes[i]
        b = suffixes[j]

        # make sure unusual forms are outliers
        if len(a) == 0 or len(b) == 0:
            return 1000

        d1 = ed.distance(a[0], b[0])

        if len(a) == 1 or len(b) == 1:
            return d1
        d2 = ed.distance(a[1], b[1]) * back_w
        avg = (d1 + d2) / 2
        return avg

    X = np.arange(len(suffixes)).reshape(-1, 1)
    clustering = dbscan(
        X,
        metric=dual_edit_metric,
        eps=0.5,  # low value to punish significant deviations
        min_samples=10,  # picked a larger value because we care about the *general* cases, not exceptions
        n_jobs=-1,
    )
    labels = list(clustering[1])

    file_cache = f"./data/unimorph/lemma_clusters/{iso}"
    out_f = open(file_cache, "w")
    for label, lemma in zip(labels, lemmas):
        out[lemma] = label
        out_f.write(f"{lemma}\t{label}\n")
    out_f.close()
    return out


def median_forms_by_lemma_cluster(unimorph_train_fp: str, pos: str) -> Dict[int, float]:
    # cluster the lemmas into categories (ideally gender or verb classes)
    # with a custom edit distance-based clustering mechanism and
    # extract the median number of word forms per lemma in each cluster

    iso = unimorph_train_fp.split("/")[-1]
    f = open(unimorph_train_fp, "r")
    lns = f.readlines()
    parsed_lns = [parse_unimorph_ln(x) for x in lns]
    # filter out wrong pos
    filtered_by_pos = [x for x in parsed_lns if x[0] == pos]

    # obtain clusters for each lemma
    lemmas = list(set([x[1] for x in filtered_by_pos]))
    cluster_map = editdistance_clustering(lemmas, iso)

    lemma_forms: DefaultDict[str, int] = defaultdict(int)
    for entry in filtered_by_pos:
        lemma = entry[1]
        lemma_forms[lemma] += 1

    # obtain median for each cluster
    num_forms_by_cluster: DefaultDict[str, List[int]] = defaultdict(list)

    for pos, lemma, token, labels in filtered_by_pos:
        cluster = cluster_map[lemma]
        num_lemma_forms = lemma_forms[lemma]
        num_forms_by_cluster[cluster].append(num_lemma_forms)

    # summarize with median
    median_by_clust: Dict[int, float] = {}

    for clust, values in num_forms_by_cluster.items():
        m = median(values)
        median_by_clust[clust] = m

    return median_by_clust


def estimate_distinct_forms(target_language: str, target_pos: str) -> int:
    # estimate how many distinct forms this specific language's POS should
    # have based on other languages, weighted by distance (lang2vec)

    # identify each language
    UNIMORPH_TRAIN_DATA = "./data/unimorph/train"

    lang_median_form: Dict[str, int] = {}
    for fp in tqdm(
        glob(f"{UNIMORPH_TRAIN_DATA}/*"),
        desc="Estimating median forms for each language...",
    ):
        iso = fp.split("/")[-1]
        med_forms = median_forms_by_lemma_cluster(fp, target_pos)
        lang_median_form[iso] = median(med_forms.values())

    print(lang_median_form)

    neigh = KNeighborsRegressor(n_neighbors=4)


estimate_distinct_forms("lat", "V")
