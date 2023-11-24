#!/bin/bash

# Build engine and launch triton server
#
# --bio and --parallel are optional, required only directory specified by --engine
# does not contain an engine, which will lead to an engine building
#
# --http-port PORT         required, triton server http port
# --repo REPO              optional, path to backend/all_models/inflight_batcher_llm,
#                          repo inside docker will be used if not present
# --engine ENGINE          required, path to tensorrt-llm engine to run, if it's not a valid engine directory, it will be built
# --model MODEL            required, path to model containing tokenizer data
# --model-name MODEL_NAME  required, model name for proxy to apply chat template, like llama-2-7b, llama-2-7b-chat-hf, mistral, zephyr,...
#                          use official name, this will make impact on Proxy(https://gitlab.deepglint.com/guoqingjiang/oaip) processing
# --devices DEVICES        optional, specify cuda devices to use, like `0,1,2,3`, please make sure there are more devices than world size of trt-engine
#                          all cuda devices would be used if absent
# --bio BIO                optional, <batch>:<input len>:<output len>, used to build engine to specify max batch-size/input-len/output-len
# --parallel PARALLEL      optional, <tp size>:<pp size>, used to build engine to specify world configuration, default: 1:1
# --qt {sq,int8kv,fp8,awq} optioal quant method for model building, fp16 engine will be built if abset
# --tmpdir TMPDIR          optional, a temp directory used to build quant engine, required if --qt is present

# bash triton_docker.sh \
#   --engine /home/gqjiang/tmpfs/test 
#   --model /home/gqjiang/tmpfs1/zephyr-7b-beta 
#   --devices 6 
#   --parallel 1:1 
#   --bio 8:2048:2048 
#   --http-port 8088 
#   --model-name zephyr

args="$@"

# docker image tag
tag="main_v0.2"

model=""
engine=""
tmpdir=""

# 处理参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --model)
            model="$2"
            shift
            shift
            ;;
        --engine)
            engine="$2"
            shift
            shift
            ;;
        --tmpdir)
            tmpdir="$2"
            shift
            shift
            ;;
        *)  # 对于未知参数，你可以选择忽略或进行适当的处理
            shift
            ;;
    esac
done

image="dockerhub.deepglint.com/lse/triton_trt_llm:$tag"


cmd="docker run --runtime=nvidia --shm-size='20g' --ipc=host --privileged "
cmd="$cmd -v /tmp/.X11-unix/:/tmp/.X11-unix/ "
cmd="$cmd -v $model:$model:ro -v $engine:$engine:rw "
if [ ! -z $tmpdir ]; then
    cmd="$cmd -v $tmpdir:$tmpdir:rw "
fi
cmd="$cmd -w /app "
cmd="$cmd -it --net=host --ulimit memlock=-1 --ulimit core=-1 --security-opt seccomp=unconfined $image "
cmd="$cmd python /app/llmtk_src/launch_triton_server.py $args"
echo $cmd

