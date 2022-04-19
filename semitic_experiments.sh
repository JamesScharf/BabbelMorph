TGT=$1



source multilingual_projection.sh heb $TGT 
source multilingual_projection.sh arz $TGT 
source multilingual_projection.sh mlt $TGT 

: '
METHOD="countback_morfessor"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="suffix_morfessor"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="suffix1"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="suffix2"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="suffix3"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="suffix4"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="suffix5"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD

METHOD="pure_morfessor"

source multilingual_embedding.sh heb $TGT $METHOD
source multilingual_embedding.sh arz $TGT $METHOD
source multilingual_embedding.sh mlt $TGT $METHOD