echo "Spanish"
source experiment.sh lat spa > /dev/null
echo "Ukrainian"
source experiment.sh rus ukr > /dev/null
echo "Icelandic"
source experiment.sh fao isl > /dev/null
echo "Arabic"
source experiment.sh heb arz > /dev/null

python3 src/utils/evaluation.py "spa;ukr;isl;arz"
