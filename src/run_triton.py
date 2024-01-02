import argparse
import os
from pathlib import Path
import json
import shutil
import torch


def query_gpu_mem(device):
    pp = os.popen(
        f"nvidia-smi --query-gpu=memory.total --id={device} --format=csv,noheader,nounits"
    )
    gb = int(pp.read()) / 1024
    pp.close()
    return gb


def determine_parallel(path: Path, devices):
    if not isinstance(devices, list):
        devices = [devices]
    gpumems = [query_gpu_mem(d) for d in devices]
    tp = 1
    pp = 0
    with open(path / "pytorch_model.bin.index.json", "r") as fp:
        sz = json.load(fp)["metadata"]["total_size"]
        gb = 1024 * 1024 * 1024
        sz = sz / gb
        total = 0
        for i, m in enumerate(gpumems):
            total += m
            if total * 0.7 > sz:
                pp = i + 1
                break
        if tp * pp == 0:
            raise RuntimeError(f"not enough memory for {sz}GB model")
    return tp, pp, devices[: tp * pp]

def run_cmd(cmd):
    s = ' '.join(cmd)
    print(f'>>> {s}')
    os.system(s)

def run_model(model="", engine="", bio="2:2048:512", name=None, devices=None, qt=None, repo=None, oaip=None, **kwargs):
    if devices is None:
        devices = [str(i) for i in range(torch.cuda.device_count())]
    if name is None:
        name = model.name
    tp, pp, devices = determine_parallel(model, devices)
    devices = ",".join(devices)
    if engine.exists():
        shutil.rmtree(engine)
    src = Path(__file__).parent
    launcher = src / "launch_triton_server.py"
    builder = src / "build_model.py"
    cmd = [
        "python",
        builder,
        "--name",
        name,
        "--src",
        model,
        "--dst",
        engine,
        "--devices",
        devices,
        "--bio",
        bio,
        "--parallel",
        f"{tp}:{pp}",
    ]
    if qt is not None:
        cmd.extend("--qt", qt)
    ret = run_cmd(cmd)
    assert ret == 0, 'build model failed'
    cmd = ["python", launcher, "--engine", engine, "--devices", devices]
    if repo:
        cmd.extend("--repo", repo)
    if oaip:
        cmd.extend("--oaip", oaip)
    ret = run_cmd(cmd)
    assert ret == 0, 'run model failed'



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
        "--qt",
        default=None,
        required=False,
        help="path to oaip, default: /app/oaip, set to `none` to disable",
    )
    parser.add_argument(
        "--oaip",
        default=None,
        required=False,
        help="path to oaip, default: /app/oaip, set to `none` to disable",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="path to backend ensemble pipeline definition",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="model name",
    )
    parser.add_argument(
        "--bio",
        type=str,
        default=None,
        help="build parameter",
    )
    # by default, user should mount the engine to /engine and run
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default=None,
        help="path to tensorrt-llm model to build",
    )
    parser.add_argument(
        "--engine",
        type=str,
        required=True,
        default=None,
        help="path to tensorrt-llm engine to run",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help="specify cuda devices to use, like `0,1,2,3`",
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    run_model(**vars(args))