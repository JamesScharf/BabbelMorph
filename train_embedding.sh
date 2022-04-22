# Train embeddings for this specific language

ISO3=$1
SEGMENT_METHOD=$2
TRAIN_TEXT_FP=./data/embedding_texts/${ISO3}_${SEGMENT_METHOD}
TRAIN_SEGMENTED=./data/segmented_embedding_texts/${ISO3}_${SEGMENT_METHOD}
MORFESSOR_MODEL=./data/morfessor_models/${ISO3}
SEGMENTED_TEXT=./data/segmented_texts/${ISO3}_${SEGMENT_METHOD}
MODEL_OUTPUT=./data/embedding_models/${ISO3}_${SEGMENT_METHOD}
MODEL_OUTPUT_VEC=./data/embedding_models/${ISO3}_${SEGMENT_METHOD}.vec
# make training text
python3 ./src/utils/prep_embed.py "${ISO3}" "${TRAIN_TEXT_FP}"

# use morfessor to segment
if test -f "$MORFESSOR_MODEL"; then
    echo "$MORFESSOR_MODEL already exists"
else
    echo "Training morfessor model..."
    morfessor -t $TRAIN_TEXT_FP -S $MORFESSOR_MODEL --lowercase --output-newlines -d ones --morph-length 2 --max-epochs 1000000
    echo "Applying morfessor model to text"
fi


# do actual model training if needed
if test -f "$MODEL_OUTPUT_VEC"; then
    echo "$MODEL_OUTPUT_VEC exists."
else
    echo "Training skipgram model"

    python3 src/utils/segment_txt.py $MORFESSOR_MODEL $TRAIN_TEXT_FP $SEGMENTED_TEXT $SEGMENT_METHOD
    ./fasttext/fasttext skipgram -input $SEGMENTED_TEXT -output $MODEL_OUTPUT -dim 50
fi