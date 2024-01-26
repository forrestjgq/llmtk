#!/bin/bash
# script to build triton+trtllm+trtllm-backend+dgtrt docker container
set -ex

root=$(dirname $(dirname $(realpath $0)))
url=$(jq -r .repo $root/docker/version.json)
tag=$(jq -r .triton $root/docker/version.json)
base=$(jq -r .base $root/docker/version.json)

pushd $root

# cleanup
sudo rm -rf $root/trtllm/build* $root/trtllm/cpp/build* $root/backend/inflight_batcher_llm/build* $root/trtllm/tensorrt_llm/libs/*

llmtk_commitid=$(git rev-parse HEAD)
pushd trtllm
trtllm_commitid=$(git rev-parse HEAD)
popd
pushd backend
backend_commitid=$(git rev-parse HEAD)
popd

extra_args=""
if [ ! -z $http_proxy ]; then
    extra_args="--build-arg proxy_val=$http_proxy"
fi

DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 $extra_args \
 --build-arg BASE_IMAGE=$url \
 --build-arg BASE_TAG=$base \
 --build-arg IMAGE=$url:$tag \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 -f $root/docker/Dockerfile.trt_llm_backend .
popd
