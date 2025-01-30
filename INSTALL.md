## Installation

Most of the requirements of this projects are similar to the ones from [https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch). If you have any issues, please first look at the issue page there.

### Requirements:
- Python >= 3.8 (mine 3.11)
- PyTorch >= 1.2 (Mine 2.2.1 (CUDA 12.1))
- torchvision >= 0.4
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly, make sure you have the latest conda
conda update --force conda

# create and activate env
conda create --name sgg_benchmark python=3.11
conda activate sgg_benchmark

# this installs the right conda dependencies for the fresh python
conda install ipython scipy h5py ninja cython matplotlib tqdm pandas

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 12.1
pip install pytorch==2.2.1 torchvision==0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# some pip dependencies
pip install -r requirements.txt

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop
