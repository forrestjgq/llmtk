#!/bin/bash
set -ex

# script to build triton+trtllm+trtllm-backend+llmtk docker container
root=$(dirname $(dirname $(realpath $0)))
url=$(jq -r .repo $root/docker/version.json)
tag=$(jq -r .app $root/docker/version.json)
base=$(jq -r .triton $root/docker/version.json)

tag="$tag-$(arch)"

pushd $root

# get commit ids
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

# prepare docker workspace
rm -rf tmp && mkdir tmp

# build oaip, it's ok to build on host and copy to image
pushd oaip
go generate ./...
popd
pushd oaip/cmd/oaip
go build -o $root/tmp/oaip .
popd

# copy files to workspace
pushd tmp
cp -r $root/script ./llmtk_script
cp -r $root/src ./llmtk_src
cp -r $root/backend/all_models .
cp -r $root/oaip/thirdparty .

extra_args=""
if [ ! -z $http_proxy ]; then
    extra_args="--build-arg proxy_val=$http_proxy"
fi

# build
DOCKER_BUILDKIT=1 docker build \
 --progress=plain \
 -t $url:$tag \
 $extra_args \
 --build-arg IMAGE=$url:$tag \
 --build-arg BASE_IMAGE=$url \
 --build-arg BASE_TAG=$base \
 --build-arg BASE_URL=$url:$base \
 --build-arg LLMTK_COMMITID=$llmtk_commitid \
 --build-arg TRTLLM_COMMITID=$trtllm_commitid \
 --build-arg BACKEND_COMMITID=$backend_commitid \
 --build-arg OAIP_COMMITID=$oaip_commitid \
 -f $root/docker/Dockerfile.trt_llm_app .
popd

rm -rf $root/tmp
