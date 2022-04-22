SRC=$1
TGT=$2


mkdir -p "./data/evaluation/${SRC}_${TGT}_whole_PROJECTOR"

PREDICTION_MODEL_EVALUATION="./data/evaluation/${SRC}_${TGT}_whole_PROJECTOR/metrics.txt"


RAW_UNIMORPH_SRC=./data/unimorph/raw/$SRC.txt
TEST_UNIMORPH_SRC=./data/unimorph/test/$SRC
VALID_UNIMORPH_SRC=./data/unimorph/valid/$SRC
TRAIN_UNIMORPH_SRC=./data/unimorph/train/$SRC
SQASHED_UNIMORPH_SRC=./data/unimorph/combined_raw/$SRC

RAW_UNIMORPH_TGT=./data/unimorph/raw/$TGT.txt
TEST_UNIMORPH_TGT=./data/unimorph/test/$TGT
VALID_UNIMORPH_TGT=./data/unimorph/valid/$TGT
TRAIN_UNIMORPH_TGT=./data/unimorph/train/$TGT
SQASHED_UNIMORPH_TGT=./data/unimorph/combined_raw/$TGT


TRAIN_DICT=./data/dictionaries/${SRC}_${TGT}_whole.dict


BITEXT=./data/staged_bitext/"$SRC"_"$TGT".prealign
FORWARD_ALIGN="./data/alignments/${SRC}_${TGT}_whole.forward"
REV_ALIGN="./data/alignments/${SRC}_${TGT}_whole.reverse"
SYM_ALIGN="./data/alignments/${SRC}_${TGT}_whole.symmetric"

FORWARD_PARAMS=./data/params/${SRC}_${TGT}_whole.forward
REV_PARAMS=./data/params/${SRC}_${TGT}_whole.rev
FORWARD_ERR=./data/alignment_errors/${SRC}_${TGT}_whole.forward
REV_ERR=./data/alignment_errors/${SRC}_${TGT}_whole.rev

# unimorph data split
if test -f "$TEST_UNIMORPH_SRC"; then 
    echo "$TEST_UNIMORPH_SRC exists" 
else
    python3 ./src/utils/data_split.py "${RAW_UNIMORPH_SRC}"  "${SQASHED_UNIMORPH_SRC}" "$TRAIN_UNIMORPH_SRC" "$VALID_UNIMORPH_SRC" "$TEST_UNIMORPH_SRC" "${SRC_MORFESSOR_MODEL}"
fi


if test -f "$TEST_UNIMORPH_TGT"; then 
    echo "$TEST_UNIMORPH_TGT exists" 
else
    python3 ./src/utils/data_split.py "${RAW_UNIMORPH_TGT}"  "${SQASHED_UNIMORPH_TGT}" "$TRAIN_UNIMORPH_TGT" "$VALID_UNIMORPH_TGT" "$TEST_UNIMORPH_TGT" "${TGT_MORFESSOR_MODEL}"
fi

echo $BITEXT
echo $FORWARD_ALIGN
echo $REV_ALIGN

if test -f "$TRAIN_DICT"; then
    echo "$TRAIN_DICT exists"
else
    python3 src/utils/make_parallel_corpus.py --fp "${BITEXT}" --src "${SRC}" --tgt "${TGT}" \
        --src_morf_model "${SRC_MORFESSOR_MODEL}" --tgt_morf_model "${TGT_MORFESSOR_MODEL}" \
        --segment_method "whole"

    ./fast_align/build/fast_align -i $BITEXT -v -d -o -I 5 -p $FORWARD_PARAMS > $FORWARD_ALIGN 
    echo "Done with forward align"
    ./fast_align/build/fast_align  -i $BITEXT -v -d -o -r -I 5 -p $REV_PARAMS > $REV_ALIGN 
    echo "Done with reverse align"
    ./fast_align/build/atools -i $FORWARD_ALIGN -j $REV_ALIGN -c intersect > $SYM_ALIGN
    echo "Done with atools intersect"
    python3 ./src/dictionary/dictionary_extraction.py \
        --align_fp $SYM_ALIGN \
        --bitext_fp $BITEXT \
        --editdist 4 \
        > $TRAIN_DICT

    echo "Wrote minimal seed dictionary to ${TRAIN_DICT}"
fi