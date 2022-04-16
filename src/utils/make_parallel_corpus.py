import argparse
import glob
from typing import List, NamedTuple, Tuple
from collections import namedtuple
from tqdm import tqdm
import morfessor
import unidecode
import morfessor_utils as mu


def find_bibles(iso: str) -> List[str]:
    return [fp for fp in glob.glob(f"./data/bible_corpus/{iso}*")]


def make_bitexts(
    out_fp: str,
    src_iso: str,
    tgt_iso: str,
    src_morph_model_fp: str,
    tgt_morph_model_fp: str,
    segment_method: str,
) -> str:
    out_f = open(out_fp, "w")
    src_bibles = find_bibles(src_iso)
    tgt_bibles = find_bibles(tgt_iso)

    # make combinations

    Bitext: NamedTuple[str, str] = namedtuple("bitext", ["src_fp", "tgt_fp"])
    combinations: List[Bitext[str, str]] = []

    io = morfessor.MorfessorIO()
    src_model = io.read_any_model(src_morph_model_fp)
    tgt_model = io.read_any_model(tgt_morph_model_fp)

    for src_fp in src_bibles:
        for tgt_fp in tgt_bibles:
            combo = Bitext(src_fp, tgt_fp)
            combinations.append(combo)

    for b in tqdm(combinations, desc=f"Making bitext for {src_iso} {tgt_iso}"):
        src_fp = b.src_fp
        tgt_fp = b.tgt_fp

        src_f = open(src_fp, "r")
        tgt_f = open(tgt_fp, "r")

        for src_ln, tgt_ln in zip(src_f, tgt_f):
            src_ln = src_ln.replace("\n", "")

            src_ln_toks = src_ln.split()
            tgt_ln_toks = tgt_ln.split()
            if len(src_ln_toks) == 0 or len(tgt_ln_toks) == 0:
                continue
            for i, tok in enumerate(src_ln_toks):
                total = mu.segment_token(src_model, tok, segment_method)
                src_ln_toks[i] = total
            src_ln = " ".join(src_ln_toks)

            for i, tok in enumerate(tgt_ln_toks):
                total = mu.segment_token(tgt_model, tok, segment_method)
                tgt_ln_toks[i] = total

            tgt_ln = " ".join(tgt_ln_toks)

            if len(src_ln.split()) < 1 or len(tgt_ln.split()) < 1:
                continue
            ln = f"{src_ln} ||| {tgt_ln}\n"
            out_f.write(ln)

        src_f.close()
        tgt_f.close()

    out_f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the bibles to src-->tgt bitext"
    )

    parser.add_argument("--fp", help="The output file")
    parser.add_argument("--src", help="The src iso")
    parser.add_argument("--tgt", help="The tgt iso")

    parser.add_argument("--src_morf_model", help="SRC trained morfessor model")
    parser.add_argument("--tgt_morf_model", help="TGT trained morfessor model")

    parser.add_argument(
        "--segment_method",
        help="One of pure_morfessor, countback_morfessor, suffix_morfessor, suffix",
    )
    args = parser.parse_args()

    out_fp = args.fp
    src_iso = args.src
    tgt_iso = args.tgt
    src_morf = args.src_morf_model
    tgt_morf = args.tgt_morf_model
    segment_method = args.segment_method

    make_bitexts(out_fp, src_iso, tgt_iso, src_morf, tgt_morf, segment_method)


if __name__ == "__main__":
    parse_args()
