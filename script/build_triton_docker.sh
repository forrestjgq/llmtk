#!/bin/bash
set -ex
# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=dockerhub.deepglint.com/lse/triton_trt_llm
tag=base-0.2

pushd $root

llmtk_commitid=$(git rev-parse HEAD)
pushd trtllm
trtllm_commitid=$(git rev-parse HEAD)
popd
pushd backend
backend_commitid=$(git rev-parse HEAD)
popd

 #--build-arg proxy_val=http://<proxy>:17892 \
DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 -f docker/Dockerfile.trt_llm_backend .
popd
