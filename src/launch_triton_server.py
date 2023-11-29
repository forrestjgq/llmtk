import argparse
import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer

from build_model import add_arguments, build

def parse_arguments():
    parser = argparse.ArgumentParser()
    # docker will build a default one inside
    parser.add_argument('--gpu-mem-fraction', type=float, default=None, help='how much GPU memory should be used, value range 0~1')
    parser.add_argument('--disable-proxy', default=False, action='store_true', help='should proxy be disabled')
    parser.add_argument('--http-port', type=int, default=8000, help='triton server/proxy http port')
    parser.add_argument('--repo', type=str, default='/app/all_models/inflight_batcher_llm', help='path to backend/all_models/inflight_batcher_llm')
    # by default, user should mount the engine to /engine and run
    parser.add_argument('--engine', type=str, default='/engine', help='path to tensorrt-llm engine to run')
    parser.add_argument('--model', type=str, default='/model', help='path to model containing tokenizer and config.json')
    parser.add_argument('--model-name', type=str, default=None, required=True, help='model name for proxy to apply chat template, like llama-2-7b, llama-2-7b-chat-hf, mistral, zephyr,...')
    parser.add_argument('--devices', type=str, default=None, help='specify cuda devices to use, like `0,1,2,3`')
    parser.add_argument('--tritonserver',
                        type=str,
                        default='/opt/tritonserver/bin/tritonserver')
    add_arguments(parser, excepts=['trtllm', 'src', 'dst', 'direct-save', 'devices'])
    return parser.parse_args()


def get_cmd(world_size, tritonserver, model_repo, http_port, devices):
    cmd = ''
    if devices and len(devices) > 0:
        cmd = 'CUDA_VISIBLE_DEVICES='+devices + ' '
    cmd += 'mpirun --allow-run-as-root '
    for i in range(world_size):
        cmd += ' -n 1 {} --allow-grpc false --grpc-port 8788 --allow-metrics false --http-port {} --model-repository={} --disable-auto-complete-config --backend-config=python,shm-region-prefix-name=prefix{}_ : '.format(
            tritonserver, http_port, model_repo, i)
    # print(cmd)
    return cmd

def replace(pbtxt, words: dict):
    with open(pbtxt, 'r') as fp:
        lines = fp.readlines()
    with open(pbtxt, 'w') as fp:
        for line in lines:
            for k, v in words.items():
                if line.find(k) >= 0:
                    line = line.replace(k, str(v))
            fp.write(line)

def detect_model_type(model):
    with open(os.path.join(model, 'config.json'), 'r') as fp:
        js = json.load(fp)
        arch = js['model_type'].lower()
        if arch == 'mistral':
            arch = 'auto'
        return arch
def read_engine_parameters(engine):
    with open(os.path.join(engine, 'config.json'), 'r') as fp:
        return json.load(fp)
def read_file(*path, flag='rb'):
    with open(os.path.join(*path), flag) as fp:
        return fp.read()

def append_pbtxt(path, d: dict):
    with open(path, 'a') as fp:
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

def build_triton_repo(repo, engine, model, model_name):
    """copy repo to /tmp and modify pbtxts"""
    assert len(model_name) > 0
    path = '/tmp/repo'
    if os.path.exists(path):
        shutil.rmtree(path)
    shutil.copytree(repo, path)
    def to(file):
        return os.path.join(path, file)
    arch = detect_model_type(model)
    param = read_engine_parameters(args.engine)

    engine_cfg = read_file(args.engine, 'config.json')
    model_cfg = read_file(args.model, 'config.json')
    model_cfg = str(base64.b64encode(model_cfg), encoding="utf8")
    engine_cfg = str(base64.b64encode(engine_cfg), encoding="utf8")

    max_batch_size = param["builder_config"]["max_batch_size"]
    replace(to('preprocessing/config.pbtxt'), {
        "${tokenizer_dir}": model,
        "${tokenizer_type}": arch,
        "${triton_max_batch_size}": max_batch_size
        })
    replace(to('postprocessing/config.pbtxt'), {
        "${tokenizer_dir}": model,
        "${tokenizer_type}": arch,
        "${triton_max_batch_size}": max_batch_size
        })
    trtllm_dict = {
        '${decoupled_mode}': 'True',
        "${batching_strategy}": 'inflight_fused_batching',
        "${engine_dir}": engine,
        "${exclude_input_in_output}": "True",
        "${triton_max_batch_size}": max_batch_size,
        "${max_queue_delay_microseconds}": 50000,
        }
    if args.gpu_mem_fraction:
        assert args.gpu_mem_fraction > 0 and args.gpu_mem_fraction < 1.0
        trtllm_dict['${kv_cache_free_gpu_mem_fraction}'] = args.gpu_mem_fraction
    replace(to('tensorrt_llm/config.pbtxt'), trtllm_dict)
    append_pbtxt(to('tensorrt_llm/config.pbtxt'), {
        "model_name": model_name,
        "max_input_len": param["builder_config"]["max_input_len"],
        "max_output_len": param["builder_config"]["max_output_len"],
        "max_batch_size": max_batch_size,
        "engine_cfg": engine_cfg,
        "model_cfg": model_cfg,
    })
        
    replace(to('ensemble/config.pbtxt'), {
        "${triton_max_batch_size}": max_batch_size
        })
    return path

def get_world_size(engine, devices):
    with open(os.path.join(engine, 'config.json'), 'r') as fp:
        js = json.load(fp)['builder_config']
        tp = js['pipeline_parallel']
        pp = js['tensor_parallel']
        world = tp * pp

        cnt = torch.cuda.device_count()
        assert cnt >= world, f'cuda device count {cnt} < world size {world}'
        if devices:
            devids = [int(s) for s in devices.split(',')]
            assert len(devids) >= world, f'specified cuda devices {devices} less than world size {world}'
        
        return world


if __name__ == '__main__':
    args = parse_arguments()
    args.repo = os.path.realpath(args.repo)
    args.model_name = args.model_name.lower()

    if not os.path.exists(os.path.join(args.engine, 'config.json')):
        # engine not present, try building new one
        build(trtllm='/app/tensorrt_llm', src=args.model, dst=args.engine, direct_save=True, **vars(args))

    model_path = build_triton_repo(args.repo, args.engine, args.model, args.model_name)
    world = get_world_size(args.engine, args.devices)
    triton_port = args.http_port
    if not args.disable_proxy:
        triton_port = 8001
        subprocess.call(f'/app/oaip -triton 127.0.0.1:{triton_port} -minloglevel 0 -logtostderr -port {args.http_port} &', shell=True)
    cmd = get_cmd(world, args.tritonserver, model_path, triton_port, args.devices)
    subprocess.call(cmd, shell=True)
