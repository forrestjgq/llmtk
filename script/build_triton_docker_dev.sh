#!/bin/bash
set -ex
# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=$(jq -r .repo $root/docker/version.json)
tag=$(jq -r .base $root/docker/version.json)

pushd $root

llmtk_commitid=$(git rev-parse HEAD)
pushd trtllm
trtllm_commitid=$(git rev-parse HEAD)
popd
pushd backend
backend_commitid=$(git rev-parse HEAD)
popd

rm -rf tmp && mkdir tmp

#prepare docker image env
pushd $root/tmp
# prepare backend requirements
mkdir backend && cp $root/backend/requirements.txt backend/
# prepare trtllm quantization requirements
mkdir quantization && cp $root/trtllm/examples/quantization/requirements.txt quantization/
# trtllm scripts
mkdir trtllm && cp $root/trtllm/docker/common/*.sh trtllm/
cp $root/script/install.sh .
popd

extra_args=""
if [ ! -z $http_proxy ]; then
    extra_args="--build-arg proxy_val=$http_proxy"
fi

DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 $extra_args \
 --build-arg IMAGE=$url:$tag \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 -f docker/Dockerfile.trt_llm_backend.env ./tmp
rm -rf tmp

popd
