# EdgeBOA

# docker image pull
docker pull kmubigdata/edge-inference:latest

# Docker container execution (Using GPU)
docker run --privileged --gpus all --shm-size 10G -it kmubigdata/edge-inference /bin/bash

# dataset, model download
chmod +x dataset_model.sh
./ dataset_model.sh

