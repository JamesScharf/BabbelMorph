SRC=$1
TGT=$2
SEGMENT_METHOD=$3

echo "Training and running: $SRC -> $TGT"

#mkdir -p "./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}"

#mkdir -p "./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}_DICTMODE"

PREDICTION_MODEL_EVALUATION="./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}/metrics.txt"
DICT_PREDICTION_MODEL_EVALUATION="./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}_DICTMODE/metrics.txt"

SRC_TRAIN_PREDICTION_MODEL_OUTPUT="./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}/${SRC}_test_predictions_src.tsv"
TGT_TRAIN_PREDICTION_MODEL_OUTPUT="./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}/${TGT}_test_predictions_tgt.tsv"

DICT_SRC_TRAIN_PREDICTION_MODEL_OUTPUT="./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}_DICTMODE/${SRC}_test_predictions_src.tsv"
DICT_TGT_TRAIN_PREDICTION_MODEL_OUTPUT="./data/evaluation/${SRC}_${TGT}_${SEGMENT_METHOD}_DICTMODE/${TGT}_test_predictions_tgt.tsv"


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

SRC_CLASSIFIER_OUT=./data/embedding_models/$SRC

SRC_EMB=./data/embedding_models/${SRC}_${SEGMENT_METHOD}.vec
TGT_EMB=./data/embedding_models/${TGT}_${SEGMENT_METHOD}.vec

BITEXT=./data/staged_bitext/"$SRC"_"$TGT".prealign

FORWARD_ALIGN="./data/alignments/${SRC}_${TGT}_${SEGMENT_METHOD}.forward"
REV_ALIGN="./data/alignments/${SRC}_${TGT}_${SEGMENT_METHOD}.reverse"
SYM_ALIGN="./data/alignments/${SRC}_${TGT}_${SEGMENT_METHOD}.symmetric"

FORWARD_PARAMS=./data/params/${SRC}_${TGT}_${SEGMENT_METHOD}.forward
REV_PARAMS=./data/params/${SRC}_${TGT}_${SEGMENT_METHOD}.rev
FORWARD_ERR=./data/alignment_errors/${SRC}_${TGT}_${SEGMENT_METHOD}.forward
REV_ERR=./data/alignment_errors/${SRC}_${TGT}_${SEGMENT_METHOD}.rev

# source to target dictionary location--partially constructed through alignment
# and edit distance
TRAIN_DICT=./data/dictionaries/${SRC}_${TGT}_${SEGMENT_METHOD}.dict


OUTPUT_FOLDER=./data/crosslingual_embeddings/"${SRC}_${TGT}_${SEGMENT_METHOD}"
mkdir -p $OUTPUT_FOLDER
SRC_MAPPED=$OUTPUT_FOLDER/$SRC.vec
TGT_MAPPED=$OUTPUT_FOLDER/$TGT.vec

SRC_MORFESSOR_MODEL=./data/morfessor_models/$SRC
TGT_MORFESSOR_MODEL=./data/morfessor_models/$TGT

# Train embeddings
source train_embedding.sh $SRC $SEGMENT_METHOD
source train_embedding.sh $TGT $SEGMENT_METHOD


if test -f "$TRAIN_DICT"; then
    echo "$TRAIN_DICT exists"
else
    python3 src/utils/make_parallel_corpus.py --fp "${BITEXT}" --src "${SRC}" --tgt "${TGT}" \
        --src_morf_model "${SRC_MORFESSOR_MODEL}" --tgt_morf_model "${TGT_MORFESSOR_MODEL}" \
        --segment_method "${SEGMENT_METHOD}"

    ./fast_align/build/fast_align -i $BITEXT -v -d -o -I 5 -p $FORWARD_PARAMS > $FORWARD_ALIGN 
    echo "Done with forward align"
    ./fast_align/build/fast_align  -i $BITEXT -v -d -o -r -I 5 -p $REV_PARAMS > $REV_ALIGN 
    echo "Done with reverse align"
    ./fast_align/build/atools -i $FORWARD_ALIGN -j $REV_ALIGN -c intersect > $SYM_ALIGN
    echo "Done with atools intersect"

    python3 ./src/dictionary/dictionary_extraction.py \
        --align_fp $SYM_ALIGN \
        --bitext_fp $BITEXT \
        > $TRAIN_DICT
    echo "Wrote minimal seed dictionary to ${TRAIN_DICT}"
fi


# output folder

if test -f "$TGT_MAPPED"; then 
    echo "$TGT_MAPPED already created"
else
    python3 ./vecmap/map_embeddings.py --semi_supervised $TRAIN_DICT $SRC_EMB $TGT_EMB $SRC_MAPPED $TGT_MAPPED --cuda -v
fi

#echo $SEGMENT_METHOD
#echo $PREDICTION_MODEL_EVALUATION

#echo "Training non-dict mode"
#if test -s "$PREDICTION_MODEL_EVALUATION"; then
#    echo "non-dict model already trained"
#else
#    python3 ./src/nearest_neighbors/classifiers.py "${SRC_MAPPED}" "${TGT_MAPPED}" "$SRC_MORFESSOR_MODEL" "$TGT_MORFESSOR_MODEL" "$TRAIN_UNIMORPH_SRC" "$VALID_UNIMORPH_SRC" "$TEST_UNIMORPH_SRC" "${SRC_TRAIN_PREDICTION_MODEL_OUTPUT}" "${SEGMENT_METHOD}" --tgt_unimorph_test_fp "$TEST_UNIMORPH_TGT" --output_tgt_unimorph_test_fp "${TGT_TRAIN_PREDICTION_MODEL_OUTPUT}" --dictionary_mode="FALSE" --dictionary_fp="" --concat "FALSE" > $PREDICTION_MODEL_EVALUATION
#fi

#echo "Training dict mode"
#if test -s "$DICT_PREDICTION_MODEL_EVALUATION"; then
#    echo "dict model already trained"
#else
#    python3 ./src/nearest_neighbors/classifiers.py "${SRC_MAPPED}" "${TGT_MAPPED}" "$SRC_MORFESSOR_MODEL" "$TGT_MORFESSOR_MODEL" "$TRAIN_UNIMORPH_SRC" "$VALID_UNIMORPH_SRC" "$TEST_UNIMORPH_SRC" "${DICT_SRC_TRAIN_PREDICTION_MODEL_OUTPUT}" "${SEGMENT_METHOD}" --tgt_unimorph_test_fp "$TEST_UNIMORPH_TGT" --output_tgt_unimorph_test_fp "${DICT_TGT_TRAIN_PREDICTION_MODEL_OUTPUT}" --dictionary_mode="TRUE" --dictionary_fp="${TRAIN_DICT}" > $DICT_PREDICTION_MODEL_EVALUATION
#fi
