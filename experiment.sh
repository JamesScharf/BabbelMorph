SRC=$1
TGT=$2


# unimorph data split

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

echo "Preparing UniMorph data splits"
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

# Prepare multilingual embeddings for model
echo "Preparing embeddings"
source multilingual_embedding.sh $SRC $TGT "pure_morfessor" > /dev/null
source multilingual_embedding.sh $SRC $TGT "countback_morfessor" > /dev/null
source multilingual_embedding.sh $SRC $TGT "suffix_morfessor" > /dev/null
source multilingual_embedding.sh $SRC $TGT "suffix1" > /dev/null
source multilingual_embedding.sh $SRC $TGT "suffix2" > /dev/null
source multilingual_embedding.sh $SRC $TGT "suffix3" > /dev/null
source multilingual_embedding.sh $SRC $TGT "suffix4" > /dev/null
source multilingual_embedding.sh $SRC $TGT "suffix5" > /dev/null

source multilingual_projection.sh $SRC $TGT
#python3 src/classifiers/alignment_project.py --fp "./data/dictionaries/${SRC}_${TGT}_whole.dict" --src "${SRC}" --tgt "${TGT}"

PRED_TEST_FP_SRC=./data/unimorph/predicted_test/${SRC}_${TGT}.src_labels
python3 src/classifiers/bootstrap_classifier.py --src "${SRC}" --tgt "${TGT}" --output "${PRED_TEST_FP_SRC}" --src_or_tgt_gen "src"

PRED_TEST_FP_TGT=./data/unimorph/predicted_test/${SRC}_${TGT}.tgt_labels
python3 src/classifiers/bootstrap_classifier.py --src "${SRC}" --tgt "${TGT}" --output "${PRED_TEST_FP_TGT}" --src_or_tgt_gen "tgt"