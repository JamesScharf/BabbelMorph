# A file for building the project

git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
cd ..
mkdir -p vecmap
git clone https://github.com/artetxem/vecmap.git

conda create --name babbel
conda activate babbel

# change line below depending on CUDA version
pip install cupy-cuda112
pip install editdistance
pip install -U ray