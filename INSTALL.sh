BASE_DIR=$(pwd)
# Download VG images
cd datasets
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

# unziping and merging
unzip images.zip
unzip images2.zip
mv VG_100K_2/* VG_100K/
rm -r VG_100K_2
rm images.zip
rm images2.zip

cd $BASE_DIR

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
conda init

source ~/.bashrc

conda update --force conda
# create and activate env
conda create --name scene_graph_benchmark python=3.11
conda activate scene_graph_benchmark

# this installs the right conda dependencies for the fresh python
conda install ipython scipy h5py ninja cython matplotlib tqdm pandas

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 12.1
conda install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# some pip dependencies
pip install -r requirements.txt

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
