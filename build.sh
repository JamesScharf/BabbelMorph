# A file for building the project

# LightGBM
pip install catboost
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension


git clone https://github.com/facebookresearch/fastText.git
cd fastText
make
cd ..
mkdir -p vecmap
git clone https://github.com/artetxem/vecmap.git

git clone https://github.com/antonisa/lang2vec.git
cd lang2vec
python3 setup.py install
cd ..

conda create --name babbel
conda activate babbel

# change line below depending on CUDA version
pip install cupy-cuda112
pip install editdistance
pip install -U ray