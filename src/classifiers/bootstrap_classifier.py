# Now that we have generated target dataset
# train a new model with predicted target labels and apply to the test set
import argparse
import text_unimorph_classifier as classifier
from typing import List


def load_test(tgt_iso: str) -> List[str]:
    # produce a list of test tokens
    f = open(f"./data/unimorph/test/{tgt_iso}")
    lns = f.readlines()
    tokens = [("tgt", x.split()[1]) for x in lns]
    return tokens


def bootstrap(src_iso: str, tgt_iso: str, out_fp: str, annotation_source: str):
    _, boot_class = classifier.make_classifier(
        src_iso, tgt_iso, train_on_generated=True, annotation_source=annotation_source
    )

    # apply to test set now
    tgt_test_tokens = load_test(tgt_iso)
    pred_labels_tgt = boot_class.predict(tgt_test_tokens)

    f = open(out_fp, "w")

    for (_, tok), labels in zip(tgt_test_tokens, pred_labels_tgt):
        labels = list(labels)
        labels.sort()
        str_labs = ";".join(labels)
        ln = f"{tok}\t{str_labs}\n"
        f.write(ln)
    f.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate potential UniMorph data for language and save in ./unimorph/generated"
    )

    parser.add_argument("--src", help="The src iso")
    parser.add_argument("--tgt", help="The tgt iso")
    parser.add_argument("--output", help="Output FP")
    parser.add_argument(
        "--src_or_tgt_gen",
        help="Whether to use src labels or tgt labels for bootstrapping. Either 'src' or 'tgt' ",
    )
    args = parser.parse_args()

    src = args.src
    tgt = args.tgt
    out_fp = args.output
    data_source = args.src_or_tgt_gen

    bootstrap(src, tgt, out_fp, data_source)


if __name__ == "__main__":
    parse_args()
