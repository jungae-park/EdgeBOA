# EdgeBOA

### Docker image pull
	docker pull kmubigdata/edge-inference:latest

### Docker container execution (Using GPU)
	docker run --privileged --gpus all --shm-size 10G -it kmubigdata/edge-inference /bin/bash

### dataset cnn model download (imagenet dataset, Mobilenet_v1,v2, Inception_v3)
	chmod +x dataset_model.sh
    ./dataset_model.sh

