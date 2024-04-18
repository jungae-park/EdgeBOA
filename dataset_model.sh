#!/bin/bash

# dataset download
mkdir imagenet
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/dataset/imagenet/imagenet_metadata.txt
curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/dataset/imagenet/imagenet_1000_raw.zip
mv imagenet_metadata.txt ./imagenet
unzip -q imagenet_1000_raw.zip -d ./imagenet && rm imagenet_1000_raw.zip

# model download
#curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v1/mobilenet_v1.zip
#unzip -q mobilenet_v1.zip && rm mobilenet_v1.zip
#curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/mobilenet_v2/mobilenet_v2.zip
#unzip -q mobilenet_v2.zip && rm mobilenet_v2.zip
#curl -O https://edge-inference.s3.us-west-2.amazonaws.com/CNN/model/inception_v3/inception_v3.zip
#unzip -q inception_v3.zip && rm inception_v3.zip
