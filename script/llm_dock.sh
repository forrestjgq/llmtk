image="dockerhub.deepglint.com/lse/triton_trt_llm:base-0.1"
name=$1

# this script will create a docker container for TensorRT-LLM development

docker run \
    --runtime=nvidia\
    --shm-size=20g\
    --ipc=host \
    --privileged \
    -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
    -v $HOME:$HOME:rw \
    -v /data:/data:rw \
    -v /mnt:/mnt:rw \
    -v /mnt2:/mnt2:rw \
    -w /$HOME/src/trtllm \
    -it --net=host --ulimit memlock=-1 --ulimit core=-1 --security-opt seccomp=unconfined \
    --name $name \
    $image \
    bash

