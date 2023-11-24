#!/bin/bash

# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=dockerhub.deepglint.com/lse/triton_trt_llm
tag=main_v0.1

pushd $root

llmtk_commitid=$(git rev-parse HEAD)
pushd trtllm
trtllm_commitid=$(git rev-parse HEAD)
popd
pushd backend
backend_commitid=$(git rev-parse HEAD)
popd

DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 --build-arg proxy_val=http://122.97.199.102:17892 \
 --build-arg llmtk_commitid=$llmtk_commitid \
 --build-arg trtllm_commitid=$trtllm_commitid \
 --build-arg backend_commitid=$backend_commitid \
 -f docker/Dockerfile.trt_llm_backend .
popd