import argparse
from cgitb import lookup

parser = argparse.ArgumentParser()
parser.add_argument("model_fp")
parser.add_argument("in_fp")
parser.add_argument("out_fp")
args = parser.parse_args()

out_fp = args.out_fp
model_fp = args.model_fp

# load model "file"
import morfessor
import morfessor_utils as mu

io = morfessor.MorfessorIO()
model = io.read_any_model(model_fp)

in_txt = open(args.in_fp, "r")
lns = in_txt.readlines()
lns = [x for x in lns if len(x.split()) != 0]

out_txt = open(out_fp, "w")

from tqdm import tqdm


def proc_ln(ln):
    splt_ln = ln.split()
    morphs = [mu.segment_token(model, t) for t in splt_ln]
    return " ".join(morphs) + "\n"


proc_lns = [proc_ln(ln) for ln in tqdm(lns)]
# merge lines
for out_ln in proc_lns:
    out_txt.write(out_ln)

out_txt.close()
