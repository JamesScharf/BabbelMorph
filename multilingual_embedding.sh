SRC=$1
TGT=$2

RAW_UNIMORPH=./data/unimorph/raw/$SRC.txt
TEST_UNIMORPH=./data/unimorph/test/$SRC
VALID_UNIMORPH=./data/unimorph/valid/$SRC
TRAIN_UNIMORPH=./data/unimorph/train/$SRC
SQASHED_UNIMORPH=./data/unimorph/combined_raw/$SRC

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
python3 ./src/utils/data_split.py "${RAW_UNIMORPH}"  "${SQASHED_UNIMORPH}" "$TRAIN_UNIMORPH" "$VALID_UNIMORPH" "$TEST_UNIMORPH" "${SRC_MORFESSOR_MODEL}"
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
#python3 ./vecmap/map_embeddings.py --semi_supervised $TRAIN_DICT $SRC_EMB $TGT_EMB $SRC_MAPPED $TGT_MAPPED --cuda -v