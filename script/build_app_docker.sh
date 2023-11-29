#!/bin/bash
set -ex
if [ $# -lt 1 ]; then
    echo "$0 <path-to-oaip>"
    exit 0
fi 

oaip=$1

# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=dockerhub.deepglint.com/lse/triton_trt_llm
base=base-0.1
tag=app-0.1

pushd $root
cp $oaip .
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
 --build-arg BASE_TAG=$base \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 -f docker/Dockerfile.trt_llm_app .
popd