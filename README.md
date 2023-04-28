# pcl-segmentation

tested with
- WSL2 Ubuntu 22.04
- python 3.10.9

requirements:
- open3d (python) with cuda 


Installation for running models with torch-points3d
- install docker
- pull image from [here](https://hub.docker.com/r/principialabs/torch-points3d/tags)
- set up nvidia container toolkit to be able to use gpus with container [see here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

how to install wsl2
- install nvidia driver on windows
- install wsl2
- install cuda toolkit for wsl2


conda create -n points3d -c conda-forge python=3.8 pytorch torchvision torchaudio pytorch-cuda=11.8 torchsparse -c pytorch -c nvidia 
pip install torch-points3d
dont't install cuda toolkit for wsl2
--> works but only partly 


Instructions for pytorch-points3d:

wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sudo sh cuda_11.1.1_455.32.00_linux.run --toolkit --silent --override

export CUDA_HOME=/usr/local/cuda-11.1

conda install openblas-devel -c anaconda
conda install python=3.8 pytorch torchvision cudatoolkit=11.1 torchsparse torch-points-kernels -c pytorch -c nvidia -c conda-forge -c torch-points3d
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"


Instructions for open3d-ml:

conda create -n open3dml
conda activate open3dml
pip install --upgrade pip
pip install -r requirements-open3d.txt
pip install open3d

sudo apt install nvidia-cudnn
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

## Use this repo

git clone git+https://github.com/cavelab-ugent/pcl-segmentation@main
pip install -e .

## Docker image

1. make sure you have the latest version of wsl (in windows command prompt: wsl --update)
2. make sure you have the latest correct windows nvidia driver 
3. install docker with docker compose inside WSL
4. install nvidia toolkit for docker inside WSL (to allow gpu usage in docker container)

cd docker
bash docker.sh