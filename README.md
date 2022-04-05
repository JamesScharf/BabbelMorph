# BabbelMorph

Steps are:

1. For source and target language (separately), train fastText embeddings with dimension 50: 
    *Pre-segment texts with morfessor, explicitly ending the word with the last morpheme.*
    *e.g., ducere --> duc__ere*

2. Obtain a minimal seed dictionary between source and target language:
    a. Obtain forward and reverse alignments.
    b. Intersect forward and reverse alignments (to maintain high precision).
    c. Obtain each word alignment, keeping only words that have low edit distance.

3. Plug fastText embeddings and seed dictionary into vecmap to obtain bilingual embeddings.