#!/bin/bash
set -ex
# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=dockerhub.deepglint.com/lse/triton_trt_llm
tag=dev-base-0.1
proxy=122.97.199.40

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

DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 --build-arg proxy_val=http://$proxy:17892 \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 -f docker/Dockerfile.trt_llm_backend.env ./tmp
rm -rf tmp

popd
