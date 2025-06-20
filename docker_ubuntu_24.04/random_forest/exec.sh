#!/bin/bash

set -e

IMAGE_NAME="fl-env"

# if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
#   echo "[DOCKER] Image '$IMAGE_NAME' not found. Building..."
#   docker build -t $IMAGE_NAME ..
# else
#   echo "[DOCKER] Image '$IMAGE_NAME' already exists. Skipping build."
# fi

#docker build -t $IMAGE_NAME ..

echo "[DOCKER] Running Random Forest container..."

docker run -it --rm \
  -p 5000:5000 \
  -v "$(pwd)":/workspace/random_forest \
  -v "$(pwd)/../data":/workspace/data \
  -v "$(pwd)/../ml_results_random_forest":/workspace/ml_results_random_forest \
  -w /workspace/random_forest \
  $IMAGE_NAME \
  bash