import argparse
import base64
import json
import os
import signal
import sys
import pathlib
import shutil
import subprocess
from pathlib import Path
import time

import torch



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        required=True,
        help="Config file",
    )
    # docker will build a default one inside
    parser.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=None,
        help="how much GPU memory should be used, value range 0~1",
    )
    parser.add_argument(
        "--oaip",
        default=None,
        required=False,
        help="path to oaip, default: /app/oaip, set to `none` to disable",
    )
    parser.add_argument(
        "--schema",
        default=None,
        required=False,
        help="image processing schema, input_feature/vision_tower/None",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="path to backend ensemble pipeline definition",
    )
    # by default, user should mount the engine to /engine and run
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="path to tensorrt-llm engine to run",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="specify cuda devices to use, like `0,1,2,3`",
    )
    parser.add_argument(
        "--tritonserver", type=str, default=None, help="specify triton server app path"
    )
    return parser.parse_args()

def get_cmd(world_size, tritonserver, model_repo, http_port, devices):
    cmd = ""
    if devices and len(devices) > 0:
        cmd = "CUDA_VISIBLE_DEVICES=" + devices + " "
    cmd += "mpirun --allow-run-as-root "
    for i in range(world_size):
        cmd += " -n 1 {} --allow-grpc false --grpc-port 8788 --allow-metrics false --http-port {} --model-repository={} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{}_ : ".format(
            tritonserver, http_port, model_repo, i
        )
    # print(cmd)
    return cmd


def replace(pbtxt, words: dict):
    with open(pbtxt, "r") as fp:
        lines = fp.readlines()
    with open(pbtxt, "w") as fp:
        for line in lines:
            for k, v in words.items():
                if line.find(k) >= 0:
                    line = line.replace(k, str(v))
            fp.write(line)


def detect_tokenizer_type(model):
    with open(os.path.join(model, "config.json"), "r") as fp:
        js = json.load(fp)
        arch = js["model_type"].lower()
        if arch in ["llama", "t5", "baichuan", "chatglm"]:
            return arch
        return "auto"


def read_engine_parameters(engine_cfg):
    with open(engine_cfg, "r") as fp:
        return json.load(fp)


def read_file(*path, flag="rb"):
    with open(os.path.join(*path), flag) as fp:
        return fp.read()


def append_pbtxt(path, d: dict):
    with open(path, "a") as fp:
        for key, value in d.items():
            txt = f"""

parameters: {{
  key: "{key}"
  value: {{
    string_value: "{value}"
  }}
}}

"""
            fp.write(txt)

def build_triton_repo(repo, engine, model, model_name, engine_cfg_path, schema, gpu_mem_fraction):
    """copy repo to /tmp and modify pbtxts

    schema: visin_tower(default): use torch vision tower, recv image and cpu
                                  decode it and use torch vision tower for feature generating
            input_feature:        oaip generates feature, save to file, give image as feature path
    """
    assert len(model_name) > 0
    path = "/tmp/repo"
    if os.path.exists(path):
        shutil.rmtree(path)
    shutil.copytree(repo, path)

    # with this model it will fail
    torm = path + '/tensorrt_llm_bls'
    if os.path.exists(torm):
        shutil.rmtree(torm)

    def to(file):
        return os.path.join(path, file)

    arch = detect_tokenizer_type(model)
    param = read_engine_parameters(engine_cfg_path)

    engine_cfg = read_file(engine_cfg_path)
    model_cfg = read_file(model, "config.json")
    model_js = json.loads(model_cfg)
    hidden = model_js["hidden_size"]
    model_cfg = str(base64.b64encode(model_cfg), encoding="utf8")
    engine_cfg = str(base64.b64encode(engine_cfg), encoding="utf8")
    max_batch_size = param["builder_config"]["max_batch_size"]

    if 'llava' in model_name.lower():
        if not schema:
            schema = "vision_tower"
        if schema == "vision_tower":
            if '34b' in model_name.lower():
                inst_cnt = 1
            else:
                inst_cnt = 2
            inst_type = 'KIND_GPU'
        elif schema == "input_feature":
            inst_cnt = 8
            inst_type = 'KIND_CPU'
        else:
            raise Exception(f"unknown schema {schema} for llava")
    else:
        if not schema:
            schema = "default"
        inst_cnt = 1
        inst_type = 'KIND_CPU'
       
        
    replace(
        to("preprocessing/config.pbtxt"),
        {
            "${tokenizer_dir}": model,
            "${tokenizer_type}": arch,
            "${triton_max_batch_size}": max_batch_size,
            "${instance_count}": inst_cnt,
            "${instance_type}": inst_type,
        },
    )
    append_pbtxt(to("preprocessing/config.pbtxt"), {
        "hidden_size": str(hidden),
        "max_input_len": param["builder_config"]["max_input_len"],
        "schema": schema,
        "model_name": model_name,
    })
    replace(
        to("postprocessing/config.pbtxt"),
        {
            "${tokenizer_dir}": model,
            "${tokenizer_type}": arch,
            "${exclude_input_in_output}": "False",
            "${triton_max_batch_size}": max_batch_size,
        },
    )
    trtllm_dict = {
        "${decoupled_mode}": "True",
        "${batching_strategy}": "inflight_fused_batching",
        "${engine_dir}": engine,
        "${exclude_input_in_output}": "False",
        "${triton_max_batch_size}": max_batch_size,
        "${max_queue_delay_microseconds}": 100,
    }
    if gpu_mem_fraction and gpu_mem_fraction > 0 and gpu_mem_fraction < 1.0:
        trtllm_dict["${kv_cache_free_gpu_mem_fraction}"] = gpu_mem_fraction
    replace(to("tensorrt_llm/config.pbtxt"), trtllm_dict)
    append_pbtxt(
        to("tensorrt_llm/config.pbtxt"),
        {
            "model_name": model_name,
            "max_input_len": param["builder_config"]["max_input_len"],
            "max_output_len": param["builder_config"]["max_output_len"],
            "max_batch_size": max_batch_size,
            "engine_cfg": engine_cfg,
            "model_cfg": model_cfg,
        },
    )

    replace(to("ensemble/config.pbtxt"), {"${triton_max_batch_size}": max_batch_size})
    return path


def get_world_size(engine_cfg, devices):
    with open(engine_cfg, "r") as fp:
        js = json.load(fp)["builder_config"]
        tp = js.get("pipeline_parallel", 1)
        pp = js.get("tensor_parallel", 1)
        world = tp * pp

        cnt = torch.cuda.device_count()
        assert cnt >= world, f"cuda device count {cnt} < world size {world}"
        if devices:
            devids = [int(s) for s in devices.split(",")]
            assert (
                len(devids) >= world
            ), f"specified cuda devices {devices} less than world size {world}"
            devids = devids[:world]
        else:
            devids = list(range(world))

        return world, devids

def decide_mem_fraction(world, devids, model_name):
    model_name = model_name.lower()
    frac = None
    if 'llava' in model_name:
        # shrink fraction so that clip could be run
        frac = 0.8
    return frac

def get_engine_cfg(engine):
    path = pathlib.Path(engine)
    if not path.exists():
        return None
    if not path.is_dir():
        raise IsADirectoryError
    found_engine = False
    cfg = None
    for p in path.iterdir():
        if p.suffix.lower() == ".engine":
            found_engine = True
        if p.parts[-1] == "config.json":
            cfg = str(p)
        if cfg is None and str(p).endswith("config.json"):
            cfg = str(p)
    print("foudn engine ", found_engine, " cfg ", cfg)
    if not found_engine:
        return None
    return cfg

def get_engine_params(engine):
    assert engine is not None
    engine = Path(engine)
    assert engine.exists() and engine.is_dir()
    param_path = engine / 'build_params.json'
    assert param_path.exists()

    with open(param_path, 'r') as fp:
        js = json.load(fp)
        js['cfg_path'] = os.path.join(engine, 'config.json')
        return js

def get_vllm_params(engine):
    assert engine is not None
    engine = Path(model)
    assert engine.exists() and engine.is_dir()
    param_path = engine / 'build_params.json'
    assert param_path.exists()

    with open(param_path, 'r') as fp:
        js = json.load(fp)
        js['cfg_path'] = os.path.join(engine, 'config.json')
        return js

def get_vllm_tp_size(devices):
        total_cnt = torch.cuda.device_count()
        assert total_cnt >= 1, f"can NOT get cuda device"
        if devices:
            devids = [int(s) for s in devices.split(",")]
            assert (
                len(devids) <= total_cnt
            ), f"specified cuda devices {devices} more than total device {total_cnt}"
            tp = len(devids)
        else:
            tp = total_cnt

        print(f'vllm tp size {tp}')
        return tp

def get_vllm_cmd(tp_size, model, address, devices, image_size, feature_size, gpu_memory_utilization):
    cmd = ""
    if devices and len(devices) > 0:
        cmd = "CUDA_VISIBLE_DEVICES=" + devices + " "

    host = address.split(':')[0]
    port = address.split(':')[1]
    input_shape_list = [1,3,image_size[0],image_size[1]]
    input_shape =  ','.join(str(x) for x in input_shape_list)

    cmd += "python -m vllm.entrypoints.openai.api_server \
                --model {} --host {} --port {} --tensor-parallel-size {} \
                --image-input-type pixel_values --image-token-id 151646 \
                --image-input-shape {} --image-feature-size {} \
                --disable-log-requests --gpu-memory-utilization {} \
                --chat-template /app/vllm/examples/template_chatml.jinja".format(
                    model,host, port, tp_size, input_shape, feature_size, gpu_memory_utilization
                )
    # print(cmd)
    return cmd

def select_best_resolution(original_size: tuple, possible_resolutions: list) -> tuple:
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    This is done by calculating the effective and wasted resolution for each possible resolution.

    The best fit resolution is the one that maximizes the effective resolution and minimizes the wasted resolution.

    Args:
        original_size (tuple):
            The original size of the image in the format (height, width).
        possible_resolutions (list):
            A list of possible resolutions in the format [(height1, width1), (height2, width2), ...].

    Returns:
        tuple: The best fit resolution in the format (height, width).
    """
    original_height, original_width = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float("inf")

    for height, width in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (
            effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution
        ):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (height, width)

    return best_fit

def divide_to_patches(image_size, patch_size: int):
    patch_cnt = 0
    height, width = image_size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patch_cnt += 1
            
    return patch_cnt

def get_llava_feature_size(model, image_size):
    model = model.strip()
    print(f'model {model}, image size {image_size}')
    config_file_path = os.path.join(model, 'config.json')
    assert os.path.exists(config_file_path) , f'{config_file_path} no config.json ?!'
    with open('/models/llava-7b-unicom-qwen2-hd-800k/config.json', 'r') as fp:
        js = json.load(fp)

    image_grid_pinpoints = (
        js['image_grid_pinpoints'] if js['image_grid_pinpoints'] is not None
        else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    )

    patch_size = 336
    preprocessor_config = os.path.join(model, 'preprocessor_config.json')
    if os.path.exists(preprocessor_config):
        with open(preprocessor_config, 'r') as fp:
            js = json.load(fp)
            patch_size = js['crop_size']


    best_resolution = select_best_resolution(image_size, image_grid_pinpoints)
    patches_cnt = divide_to_patches(best_resolution, patch_size)
    print(f'patch cnt: {patches_cnt}')
    feature_size = (1 + patches_cnt) * 576
    return feature_size

class Args:
    def __init__(self) -> None:
        pass
    def add(self, key, value):
        setattr(self, key, value)

def main(args):
    processes = {}
    # startup oaip if required, before triton startup
    oaip = args.oaip if args.oaip else "/app/oaip"
    if oaip != 'none':
        assert os.path.exists(oaip), f"oaip not found: {oaip}"
        oaip = f"{oaip} -config {args.config}"
        print(">>> ", oaip)
        processes['oaip'] = subprocess.Popen(oaip, shell=True, preexec_fn=os.setsid)

    if args.engine_name.lower() == "triton":
        params = get_engine_params(args.engine)
        model_name = params['name'].lower()
        engine_cfg = params['cfg_path']

        # check engine configuration
        model = os.path.join(args.engine, 'model')
        assert os.path.exists(model)

        if not args.repo:
            args.repo = "/app/all_models/inflight_batcher_llm"
        args.repo = os.path.realpath(args.repo)
        assert os.path.exists(args.repo), f"repo {args.repo} not exist"

        # prepare triton parameters
        triton = args.tritonserver if args.tritonserver else "/opt/tritonserver/bin/tritonserver"
        triton_port = args.address.split(':')[1]
        world, devids = get_world_size(engine_cfg, args.devices)

        if args.gpu_mem_fraction is None or args.gpu_mem_fraction < 0.1:
            args.gpu_mem_fraction  = decide_mem_fraction(world, devids, model_name)
        model_path = build_triton_repo(
            args.repo, args.engine, model, model_name, engine_cfg, args.schema, args.gpu_mem_fraction
        )
        cmd = get_cmd(world, triton, model_path, triton_port, args.devices)
        print(">>> ", cmd)
        processes['triton'] = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    elif args.engine_name.lower() == "vllm":
        #set timeout for vllm
        if args.timeout is not None and args.timeout > 60: 
            os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = str(args.timeout)

        feature_size = get_llava_feature_size(args.model, args.process_image_size)
        tp_size = get_vllm_tp_size(args.devices)
        gpu_mem_fraction = args.gpu_mem_fraction

        cmd = get_vllm_cmd(tp_size, args.model, args.address, args.devices, args.process_image_size, feature_size, gpu_mem_fraction)
        print(">>> ", cmd)
        processes['vllm'] = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    else:
        assert 0, "engine name must be triton or vllm"

    # this program monitor triton/vllm and oaip and any processes started by myself
    # any exit subprocess will terminate everything
    exits = False
    try:
        while not exits:
            for k, v in processes.items():
                ret = v.poll()
                if ret is not None:
                    print(f'subprocess {k} exited: {ret}')
                    exits = True
            time.sleep(3)
    except Exception as e:
        print(e)
        print('now terminating everything')

    for k, v in processes.items():
        ret = v.poll()
        if ret is None:
            print(f'terminating {k}, pid {v.pid}')
            try:
                os.killpg(v.pid,signal.SIGTERM) 
            except Exception as e:
                print(e)
            print(f'stop terminating')
    

def set_value(js, args, name, defval, *aargs):
    """if args.name is None, load from js[*args], if still not present, use defval
    """
    v = getattr(args, name, None)
    if v is not None:
        return
    if len(aargs) > 0:
        v = js
        for param in aargs:
            v = v.get(param, None)
            if v is None:
                break
    if v is None:
        v = defval
    setattr(args, name, v)
    
if __name__ == "__main__":
    print(f">> cmd line: {' '.join(sys.argv)}")
    args = parse_arguments()
    assert len(args.config) > 0 and os.path.exists(args.config)
    with open(args.config, 'r') as fp:
        js = json.load(fp)
    def setval(*a):
        set_value(js, args, *a)
    setval("oaip", None, "sys", "oaip")
    setval("devices", None, "sys", "devices")
    setval("oaip", None, "sys", "oaip")
    setval("engine_name", None, "engine", "name")

    engine_name = js['engine']['name']
    if engine_name == "vllm":
        #vllm args
        print(f'select vllm engine')
        setval("address", None, "engine", "vllm", "address")
        setval("timeout", 60, "engine", "vllm", "timeout")
        setval("model", None, "engine", "vllm", "model")
        setval("gpu_mem_fraction", 0.9, "engine", "vllm", "gpuMemFraction")
        setval("process_image_size", [1920, 1080], "engine", "vllm", "process_image_size")
        assert args.model is not None and len(args.model) > 0
    elif engine_name == "triton":
        #triton args
        print(f'select triton engine')
        setval("schema", None, "engine", "triton", "schema")
        setval("repo", None, "engine", "triton", "repo")
        setval("engine", None, "engine", "triton", "engine")
        setval("tritonserver", None, "engine", "triton", "tritonServer")
        setval("address", None, "engine", "triton", "address")
        setval("gpu_mem_fraction", None, "engine", "triton", "gpuMemFraction")
    else:
        assert 0, "only support triton & vllm engine"

    assert args.address is not None and len(args.address) > 0

    main(args)
    
