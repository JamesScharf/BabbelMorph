TGT=$1

source multilingual_projection.sh rus $TGT
source multilingual_projection.sh ukr $TGT
source multilingual_projection.sh mkd $TGT
source multilingual_projection.sh slk $TGT
source multilingual_projection.sh slv $TGT
source multilingual_projection.sh bul $TGT

: '
METHOD="countback_morfessor"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="suffix_morfessor"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="suffix1"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="suffix2"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="suffix3"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="suffix4"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="suffix5"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD

METHOD="pure_morfessor"
source multilingual_embedding.sh rus $TGT $METHOD
source multilingual_embedding.sh ukr $TGT $METHOD
source multilingual_embedding.sh mkd $TGT $METHOD
source multilingual_embedding.sh slk $TGT $METHOD
source multilingual_embedding.sh slv $TGT $METHOD
source multilingual_embedding.sh bul $TGT $METHOD