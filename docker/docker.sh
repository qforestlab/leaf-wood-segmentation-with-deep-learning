#!/bin/bash
#
# Shell commmands to start docker container and attach a bash shell to it
# -d flag makes the container run in the background
# --build flag builds the container if it wasn't already

docker compose up -d --build;
docker compose exec open3dml git clone https://github.com/woutervdbroeck/Open3D-ML;
docker compose exec open3dml pip install -e ./Open3D-ML;
docker compose exec open3dml pip install git+https://github.com/qforestlab/leaf-wood-segmentation-with-deep-learning;
docker compose exec open3dml bash