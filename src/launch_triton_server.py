import argparse
import base64
import json
import os
import pathlib
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoTokenizer

from build_model import add_arguments, build


def parse_arguments():
    parser = argparse.ArgumentParser()
    # docker will build a default one inside
    parser.add_argument(
        "--gpu-mem-fraction",
        type=float,
        default=None,
        help="how much GPU memory should be used, value range 0~1",
    )
    parser.add_argument(
        "--oaip",
        default="/app/oaip",
        required=False,
        help="path to oaip, default: /app/oaip",
    )
    parser.add_argument(
        "--schema",
        default=None,
        required=False,
        help="image processing schema, input_feature/vision_tower/None",
    )
    parser.add_argument(
        "--disable-proxy",
        default=False,
        action="store_true",
        help="should proxy be disabled",
    )
    parser.add_argument(
        "--http-port", type=int, default=8000, help="triton server/proxy http port"
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
        default="/engine",
        help="path to tensorrt-llm engine to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/model",
        help="path to model containing tokenizer and config.json",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        required=False,
        help="model name for proxy to apply chat template, like llama-2-7b, llama-2-7b-chat-hf, mistral, zephyr,..., if not present, use directory name of --src instead",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="specify cuda devices to use, like `0,1,2,3`",
    )
    parser.add_argument(
        "--tritonserver", type=str, default="/opt/tritonserver/bin/tritonserver"
    )
    parser.add_argument(
        "--trtllm",
        type=str,
        default="/app/tensorrt_llm",
        help="specify trtllm directory",
    )
    add_arguments(
        parser, excepts=["trtllm", "src", "dst", "direct-save", "devices", "name"]
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

def build_triton_repo(repo, engine, model, model_name, engine_cfg_path, schema):
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

    def to(file):
        return os.path.join(path, file)

    arch = detect_tokenizer_type(model)
    param = read_engine_parameters(engine_cfg_path)

    engine_cfg = read_file(engine_cfg_path)
    model_cfg = read_file(args.model, "config.json")
    model_js = json.loads(model_cfg)
    hidden = model_js["hidden_size"]
    model_cfg = str(base64.b64encode(model_cfg), encoding="utf8")
    engine_cfg = str(base64.b64encode(engine_cfg), encoding="utf8")
    max_batch_size = param["builder_config"]["max_batch_size"]

    if 'llava' in model_name.lower():
        if schema is None:
            schema = "vision_tower"
        if schema == "vision_tower":
            inst_cnt = 2
            inst_type = 'KIND_GPU'
        elif schema == "input_feature":
            inst_cnt = 8
            inst_type = 'KIND_CPU'
        else:
            raise Exception(f"unknown schema {schema} for llava")
    else:
        if schema is None:
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
        "schema": schema,
    })
    replace(
        to("postprocessing/config.pbtxt"),
        {
            "${tokenizer_dir}": model,
            "${tokenizer_type}": arch,
            "${triton_max_batch_size}": max_batch_size,
        },
    )
    trtllm_dict = {
        "${decoupled_mode}": "True",
        "${batching_strategy}": "inflight_fused_batching",
        "${engine_dir}": engine,
        "${exclude_input_in_output}": "True",
        "${triton_max_batch_size}": max_batch_size,
        "${max_queue_delay_microseconds}": 50000,
    }
    if args.gpu_mem_fraction:
        assert args.gpu_mem_fraction > 0 and args.gpu_mem_fraction < 1.0
        trtllm_dict["${kv_cache_free_gpu_mem_fraction}"] = args.gpu_mem_fraction
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

        return world


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


if __name__ == "__main__":
    args = parse_arguments()
    if args.repo is not None:
        args.repo = os.path.realpath(args.repo)
    if args.model_name is None:
        args.model_name = Path(args.model).parts[-1]
    args.model_name = args.model_name.lower()

    cfg = get_engine_cfg(args.engine)
    if cfg is None:
        # engine not present, try building new one
        build(
            src=args.model,
            dst=args.engine,
            direct_save=True,
            name=args.model_name,
            **vars(args),
        )
        cfg = get_engine_cfg(args.engine)

    # check engine configuration
    assert cfg is not None, "engine config not found"
    args.model = os.path.join(args.engine, 'model')
    assert os.path.exists(args.model)

    filepath = Path(cfg)
    assert filepath.exists(), f"engine config {filepath} not found"
    if filepath.parts[-1] != "config.json":
        # trtllm sucks on chatglm model building because they write chatglmxxx-config.json
        # instead of config.json, which can not be recognized by backend
        if args.model_name.startswith("chatglm"):
            shutil.copyfile(filepath, filepath.parent / "config.json")
        else:
            raise Exception("lack of impl")

    if args.repo is None:
        if "llava" in args.model_name:
            args.repo = "/app/all_models/llava"
        else:
            args.repo = "/app/all_models/inflight_batcher_llm"
    model_path = build_triton_repo(
        args.repo, args.engine, args.model, args.model_name, cfg, args.schema
    )
    world = get_world_size(cfg, args.devices)
    triton_port = args.http_port
    if not args.disable_proxy:
        triton_port += 1 # triton port is oaip port + 1
        assert os.path.exists(args.oaip), f"oaip not found: {args.oaip}"
        oaip = f"{args.oaip} -triton 127.0.0.1:{triton_port} -http-log-level 10 -minloglevel 1 -logtostderr -port {args.http_port} -tmpdir /data/jgq/tmpfs/b64 &"
        print(">>> ", oaip)
        subprocess.call(oaip, shell=True)
    cmd = get_cmd(world, args.tritonserver, model_path, triton_port, args.devices)
    subprocess.call(cmd, shell=True)
