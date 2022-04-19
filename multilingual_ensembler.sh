SRC=$1
METHOD=$2 # threshold3, plurality, majority, union
mkdir -p "./data/evaluation/all_${SRC}_${METHOD}"
OUT_DATA="./data/evaluation/all_${SRC}_${METHOD}/metrics.txt"
OUT_PREDS="./data/evaluation/all_${SRC}_${METHOD}/${SRC}_test_predictions.tsv"

python3 src/ensemble_vote/compute_election.py "$SRC" "$METHOD" "$OUT_PREDS" > $OUT_DATA