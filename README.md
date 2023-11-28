# pcl-segmentation

## Setup repo with conda environment

1. install anaconda/miniconda and make a new environment:
```
conda create -n leafwood python==3.8
conda activate leafwood
```
2. install required packages:
```
conda install pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install open3d==0.17.0 scikit-learn==1.2.1 tensorboard
```
3. Install fork of open3D-ml
```
mkdir ~/repo
cd ~/repo
git clone https://github.com/woutervdbroeck/Open3D-ML
pip install -e .
```
4. Install this package
```
mkdir ~/project
cd ~/project
git clone git+https://github.com/cavelab-ugent/pcl-segmentation@main
cd pcl-segmentation
pip install -e .
```

## Setup repo with Docker 

1. make sure you have the latest version of wsl (in windows command prompt: wsl --update)
2. make sure you have the latest correct windows nvidia driver 
3. install docker with docker compose inside WSL
4. install nvidia toolkit for docker inside WSL (to allow gpu usage in docker container) [see here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
5. check your data path in the docker-compose.yml file (under -volumes)
5. start docker image:
```
cd ~/project/pcl-segmentation/docker
bash docker.sh
```

## Using this repo

Everything should be specified in a a yammel config file in the cfg directory.

Example structure of files:

dataset_path
  |- train_dir
  |- val_dir
  |- test_dir: in case of inference files should be placed inside this folder
      |- file_name.npy
  |- test_result_folder: predictions will be saved in this folder

run training: ```python scripts/train.py cfg/name_of_config_file.yml```

run inference on new data: ```python scripts/infer.py cfg/name_of_config_file.yml```

