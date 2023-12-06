#!/bin/bash
set -ex
url=dockerhub.deepglint.com/lse/triton_trt_llm

model_name=""
model=""
engine=""
device=0
app=app-0.3
port=8000
bio=""
opts=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --name) model_name="$2"; shift; shift ;;
        --model) model="$2"; shift; shift ;;
        --engine) engine="$2"; shift; shift ;;
        --device) device="$2"; shift; shift ;;
        --app) app="$2"; shift; shift ;;
        --port) port="$2"; shift; shift ;;
        --bio) bio="$2"; shift; shift ;;
        --disable-proxy) opts="$opts --disable-proxy"; shift ;;
        --gpu-mem-fraction) opts="$opts --gpu-mem-fraction $2"; shift ; shift ;;
        *) 
        echo "unknown option $1"

        cat <<END
$0 --name <model-name> --model <model-path> --engine <engine-path>
   [--device <CUDA-devices>] [--app <app-docker-tag>] [--port <http-port>]
   [--bio <batch:input-len:output-len>] [--disable-proxy]

   Start app for LLM service.

    --name              model official name, like Llama-2-7b-chat, zephyr-beta...
    --model             model path
    --engine            engine path, if it contains a config.json, it will be treated as a valid
                        trt-engine and will be loaded by triton directly, otherwise it should
                        be writtable and triton will convert from <model> and write into this 
                        directory, in this case <bio> option is required
    --device            CUDA device to use, default 0, could be set with several devices like 2,3,5,6
    --app               docker image tag to use to startup, default app-0.1
    --port              service port exposed by docker to app http service
    --bio               parameters to convert model to engine, only be used and required if <engine> 
                        does not contain a config.json file. 
                        Argument format: batch:input-len:output-len for max batch size, max
                        input length, max output length.
                        For example, --bio 8:2048:2048
    --disable-proxy     app will start triton inference server internally along with an proxy
                        to provide OpenAI API service through HTTP. If this option is provided,
                        OpenAI API service will not be started and <port> will be directed to
                        triton inference service through HTTP.
    --gpu-mem-fraction  how much GPU memory should be used at most, value range 0.0~1.0(less than 1.0)
                        
    
END
        exit 1
        ;;
    esac
done

if [ -z $model_name]; then
    echo "model_name not present"
    exit 1
fi
if [ -z $model]; then
    echo "model not present"
    exit 1
fi
if [ ! -e $model]; then
    echo "model $model not found"
    exit 1
fi
if [ -z $engine]; then
    echo "engine not present"
    exit 1
fi
if [ ! -e $engine]; then
    echo "engine $engine not found"
    exit 1
fi

if [ ! -z $bio ]; then
    opts="$opts --bio $bio"
fi

docker run -it --rm --runtime=nvidia -p $port:8000 \
    --shm-size='20g' --ipc=host --privileged \
    --ulimit memlock=-1 --ulimit core=-1 --security-opt seccomp=unconfined \
    -v $model:$model:ro \
    -v $engine:$engine:rw \
    $url:$app \
    python /app/llmtk_src/launch_triton_server.py \
        --http-port 8000 \
        --engine $engine \
        --model $model \
        --model-name $model_name \
        $opts \
        --devices $device