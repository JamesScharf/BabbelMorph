# Train embeddings for this specific language

ISO3=$1
TRAIN_TEXT_FP=./data/embedding_texts/$ISO3
TRAIN_SEGMENTED=./data/segmented_embedding_texts/$ISO3
MORFESSOR_MODEL=./data/morfessor_models/$ISO3
SEGMENTED_TEXT=./data/segmented_texts/$ISO3
MODEL_OUTPUT=./data/embedding_models/$ISO3
MODEL_OUTPUT_VEC=./data/embedding_models/$ISO3.vec
# make training text
python3 ./src/utils/prep_embed.py "${ISO3}" "${TRAIN_TEXT_FP}"

# use morfessor to segment
#if test -f "$MORFESSOR_MODEL"; then
#    echo "$MORFESSOR_MODEL already exists"
#else
    echo "Training morfessor model..."
    morfessor -t $TRAIN_TEXT_FP -S $MORFESSOR_MODEL --lowercase --output-newlines -d ones --morph-length 2
#fi

# segment text
#echo "Applying morfessor model to text..."
python3 src/utils/segment_txt.py $MORFESSOR_MODEL $TRAIN_TEXT_FP $SEGMENTED_TEXT


# do actual model training if needed
#if test -f "$MODEL_OUTPUT_VEC"; then
#    echo "$MODEL_OUTPUT_VEC exists."
#else
    echo "Training skipgram model"
    ./fasttext/fasttext skipgram -input $SEGMENTED_TEXT -output $MODEL_OUTPUT -dim 50
#fi