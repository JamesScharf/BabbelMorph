SRC=$1
TGT=$2

mkdir -p "./data/evaluation/${SRC}_${TGT}"

PREDICTION_MODEL_EVALUATION="./data/evaluation/${SRC}_${TGT}/metrics.txt"
SRC_TRAIN_PREDICTION_MODEL_OUTPUT="./data/evaluation/${SRC}_${TGT}/${SRC}_test_predictions_src.tsv"
TGT_TRAIN_PREDICTION_MODEL_OUTPUT="./data/evaluation/${SRC}_${TGT}/${TGT}_test_predictions_tgt.tsv"


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

SRC_EMB=./data/embedding_models/$SRC.vec
TGT_EMB=./data/embedding_models/$TGT.vec

BITEXT=./data/staged_bitext/"$SRC"_"$TGT".prealign

FORWARD_ALIGN="./data/alignments/${SRC}_${TGT}.forward"
REV_ALIGN="./data/alignments/${SRC}_${TGT}.reverse"
SYM_ALIGN="./data/alignments/${SRC}_${TGT}.symmetric"

FORWARD_PARAMS=./data/params/${SRC}_${TGT}.forward
REV_PARAMS=./data/params/${SRC}_${TGT}.rev
FORWARD_ERR=./data/alignment_errors/${SRC}_${TGT}.forward
REV_ERR=./data/alignment_errors/${SRC}_${TGT}.rev

# source to target dictionary location--partially constructed through alignment
# and edit distance
TRAIN_DICT=./data/dictionaries/${SRC}_${TGT}.dict


SRC_MAPPED=$OUTPUT_FOLDER/$SRC.vec
TGT_MAPPED=$OUTPUT_FOLDER/$TGT.vec

SRC_MORFESSOR_MODEL=./data/morfessor_models/$SRC
TGT_MORFESSOR_MODEL=./data/morfessor_models/$TGT

# Train embeddings
#source train_embedding.sh $SRC
#source train_embedding.sh $TGT


# unimorph data split
python3 ./src/utils/data_split.py "${RAW_UNIMORPH_SRC}"  "${SQASHED_UNIMORPH_SRC}" "$TRAIN_UNIMORPH_SRC" "$VALID_UNIMORPH_SRC" "$TEST_UNIMORPH_SRC" "${SRC_MORFESSOR_MODEL}"
python3 ./src/utils/data_split.py "${RAW_UNIMORPH_TGT}"  "${SQASHED_UNIMORPH_TGT}" "$TRAIN_UNIMORPH_TGT" "$VALID_UNIMORPH_TGT" "$TEST_UNIMORPH_TGT" "${TGT_MORFESSOR_MODEL}"

#python3 src/utils/make_parallel_corpus.py --fp "${BITEXT}" --src "${SRC}" --tgt "${TGT}" \
#    --src_morf_model "${SRC_MORFESSOR_MODEL}" --tgt_morf_model "${TGT_MORFESSOR_MODEL}"
#./fast_align/build/fast_align -i $BITEXT -v -d -o -I 5 -p $FORWARD_PARAMS > $FORWARD_ALIGN 
echo "Done with forward align"
#./fast_align/build/fast_align  -i $BITEXT -v -d -o -r -I 5 -p $REV_PARAMS > $REV_ALIGN 
echo "Done with reverse align"
#./fast_align/build/atools -i $FORWARD_ALIGN -j $REV_ALIGN -c intersect > $SYM_ALIGN
echo "Done with atools intersect"

#python3 ./src/dictionary/dictionary_extraction.py \
#    --align_fp $SYM_ALIGN \
#    --bitext_fp $BITEXT \
#    > $TRAIN_DICT
echo "Wrote minimal seed dictionary to ${TRAIN_DICT}"


# output folder
OUTPUT_FOLDER=./data/crosslingual_embeddings/"${SRC}_${TGT}"
mkdir -p $OUTPUT_FOLDER
mkdir -p ./data/crosslingual_embeddings/${SRC}_${TGT}
#python3 ./vecmap/map_embeddings.py --semi_supervised $TRAIN_DICT $SRC_EMB $TGT_EMB $SRC_MAPPED $TGT_MAPPED --cuda -v

mkdir -p ./data/crosslingual_token_embeddings/"${SRC}_${TGT}"/
CROSSLINGUAL_SRC_TOKEN_EMBEDDINGS=./data/crosslingual_token_embeddings/"${SRC}_${TGT}"/$SRC
CROSSLINGUAL_TGT_TOKEN_EMBEDDINGS=./data/crosslingual_token_embeddings/"${SRC}_${TGT}"/$TGT

CROSSLINGUAL_SRC_SUFFIX_EMBEDDINGS=${OUTPUT_FOLDER}/${SRC}.vec

CROSSLINGUAL_TGT_SUFFIX_EMBEDDINGS=${OUTPUT_FOLDER}/${TGT}.vec
#python3 ./src/nearest_neighbors/reconstruct_token_vec.py \
#    "${OUTPUT_FOLDER}/${SRC}.vec" "${SRC}" "${SRC_MORFESSOR_MODEL}" \
#    >  $CROSSLINGUAL_SRC_TOKEN_EMBEDDINGS

#python3 ./src/nearest_neighbors/reconstruct_token_vec.py \
#    "${OUTPUT_FOLDER}/${TGT}.vec" "${TGT}" "${TGT_MORFESSOR_MODEL}" \
#    >  $CROSSLINGUAL_TGT_TOKEN_EMBEDDINGS

echo "Converted crosslingual embeddings to token (as opposed to suffix) format"
#python3 ./src/nearest_neighbors/interactive_nn.py "${CROSSLINGUAL_SRC_TOKEN_EMBEDDINGS}" "${CROSSLINGUAL_TGT_TOKEN_EMBEDDINGS}" "tgt"

python3 ./src/nearest_neighbors/classifiers.py "${CROSSLINGUAL_SRC_SUFFIX_EMBEDDINGS}" "${CROSSLINGUAL_TGT_SUFFIX_EMBEDDINGS}" "$SRC_MORFESSOR_MODEL" "$TGT_MORFESSOR_MODEL" "$TRAIN_UNIMORPH_SRC" "$VALID_UNIMORPH_SRC" "$TEST_UNIMORPH_SRC" "${SRC_TRAIN_PREDICTION_MODEL_OUTPUT}" --tgt_unimorph_test_fp "$TEST_UNIMORPH_TGT" --output_tgt_unimorph_test_fp "${TGT_TRAIN_PREDICTION_MODEL_OUTPUT}"  > $PREDICTION_MODEL_EVALUATION

