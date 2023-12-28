#!/bin/bash
set -ex

# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=$(jq .repo $root/docker/version.json)
tag=$(jq .app $root/docker/version.json)
base=$(jq .triton $root/docker/version.json)

pushd $root

rm -rf tmp && mkdir tmp

# build oaip, it's ok to build on host and copy to image
pushd oaip/cmd/oaip
go build -o $root/tmp/oaip .
popd

cp -r $root/script .
cp -r $root/src .
cp -r $root/backend/all_models .
cp $root/requirements.txt .

llmtk_commitid=$(git rev-parse HEAD)
pushd trtllm
trtllm_commitid=$(git rev-parse HEAD)
popd
pushd backend
backend_commitid=$(git rev-parse HEAD)
popd
pushd oaip
oaip_commitid=$(git rev-parse HEAD)
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
 --build-arg BASE_IMAGE=$url \
 --build-arg BASE_TAG=$base \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 --build-arg OAIP_COMMITID=$oaip_commitid \
 -f docker/Dockerfile.trt_llm_app .
popd