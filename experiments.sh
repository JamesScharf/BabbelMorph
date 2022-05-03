#echo "Spanish"
#source experiment.sh lat spa > /dev/null
source experiment.sh ron spa > /dev/null

#echo "Ukrainian"
source experiment.sh rus ukr > /dev/null
#source experiment.sh tur kaz > /dev/null

source experiment.sh urd hin > /dev/null

source experiment.sh ell grc > /dev/null

source experiment.sh isl swe > /dev/null
source experiment.sh nno swe > /dev/null

#echo "Maltese"
source experiment.sh heb mlt > /dev/null

#source experiment.sh tur crh > /dev/null

source experiment.sh mkd ukr /dev/null

source experiment.sh lav lit

source experiment.sh  nob nno

#python3 src/utils/evaluation.py "spa;ukr;isl;arz" > experiment_metrics.tsv
