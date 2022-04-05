import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("iso3")
parser.add_argument("out_fp")
args = parser.parse_args()
iso3 = args.iso3
out_fp = args.out_fp

# find all corpora that begin with ISO code
out_f = open(out_fp, "w")
out_f.write("")
out_f.close()
out_f = open(out_fp, "a")
for fp in glob.glob("./data/bible_corpus/*"):
    back = fp.split("/")[-1]
    iso = back.split("-")[0]

    if iso == iso3:
        curr_f = open(fp, "r")
        data = curr_f.read()
        out_f.write(data)

out_f.close()
