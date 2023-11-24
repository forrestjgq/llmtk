import os
import sys
import json
import shutil
from itertools import product
import multiprocessing as mp
from build_model import build, add_arguments
import argparse
from typing import List, Optional

g_moddefs=[
    [16,256,256],
    [16,256,512],
    [16,256,1024],
    [16,512,512],
    [16,512,1024],
    [32,256,256],
    [32,256,512],
    [32,256,1024],
    [32,512,512],
    [32,512,1024],
    [128,512,512],
    [128,512,1024],
    [128,512,2048],
    [128,1024,1024],
    [128,1024,2048],
    [128,2048,2048],
]
# g_batchs=[1, 16, 64, 128]
# g_inputs=[ 256, 1024, 2048]
# g_outputs=[256, 1024, 2048]
# g_batchs=[1, 8, 16, 32, 128]
g_batchs=None
g_inputs=[64, 256, 512]
g_outputs = [64, 256, 512, 1024, 2048]
# worlds contains composition of tp and pp, each item in list is formated as [tp, pp]
# g_worlds=[[1,1], [2, 1], [2, 2]]
g_worlds=[[1,1]]

def abs_path(file):
    root = os.path.dirname(os.path.dirname(__file__))
    return os.path.join(root, file)
def subprocess(q, **kwargs):
    ret = build(**kwargs)
    q.put(ret)
def build_model(batch, input, output, tp, pp, **kwargs):
    bio = f'{batch}:{input}:{output}'
    parallel = f'{tp}:{pp}'
    q = mp.Queue()
    args = kwargs.copy()
    args["bio"] = bio
    args["parallel"] = parallel
    p = mp.Process(target=subprocess, args=(q,), kwargs=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        return None, None
    return q.get()

"""-m, --model arg              Model name specified for engines. (default:
        gpt_350m)
    --engine_dir arg         Directory that store the engines.
    --batch_size arg         Specify batch size(s) you want to benchmark.
                            Multiple batch sizes can be separated by
                            ";", example: "1;8;64". (default: 8)
    --input_output_len arg   Specify input-output length(s) you want to
                            benchmark. Multiple input lengths can be
                            separated by ";", example: "60,20;128,20".
                            (default: 128,20)
    --log_level arg          Choose log level between
                            verbose/info/warning/error/internal_error.
                            (default: error)
    --warm_up arg            Specify warm up iterations before benchmark
                            starts. (default: 2)
    --num_runs arg           Minimal number of iterations to run during
                            benchmarking. (default: 10)
    --duration arg           Minimal duration of iterations to measure in
                            seconds. (default: 60)
    --num_micro_batches arg  Number of micro batches if enabling pipeline
                            parallelism.
    --enable_cuda_graph      Execute GPT session with CUDA graph.
"""
def make_common(cmd, js, model: str, devices: Optional[List[str]]):
    global g_batchs 
    global g_inputs 
    global g_outputs 
    max_batch_size=js["builder_config"]["max_batch_size"]
    max_input_len=js["builder_config"]["max_input_len"]
    max_output_len=js["builder_config"]["max_output_len"]
    tp=js["builder_config"]["tensor_parallel"]
    pp=js["builder_config"]["pipeline_parallel"]
    if g_batchs is None:
        g_batchs = [max_batch_size]
    batchs=[str(b) for b in g_batchs if b <= max_batch_size]
    inputs=[str(i) for i in g_inputs if i <= max_input_len]
    outputs=[str(i) for i in g_outputs if i <= max_output_len]
    batchstr=';'.join(batchs)
    p=list(product(inputs, outputs))
    ios=';'.join([','.join(t) for t in p])

    world = tp * pp
    if devices is None:
        devices=','.join([str(i) for i in range(world)])
    else:
        assert len(devices) >= world
        devices = ','.join(devices[:world]) 
    cmds=[]
    print("")
    print(f">>>> max batch {max_batch_size} in {max_input_len} out {max_output_len} tp {tp} pp {pp}")
    def ap(s):
        nonlocal cmds
        cmds.append(s.strip())
    ap(f'CUDA_VISIBLE_DEVICES="{devices}" ')
    if world > 1:
        ap( f'mpirun -n {world} --allow-run-as-root ')
    ap(cmd)
    ap(f'--engine_dir={model}')
    ap(f'--batch_size "{batchstr}" ')
    ap(f'--input_output_len "{ios}" ')
    return cmds

def make_python(js, model: str, devices: Optional[List[str]]):
    max_batch_size=js["builder_config"]["max_batch_size"]
    max_input_len=js["builder_config"]["max_input_len"]
    max_output_len=js["builder_config"]["max_output_len"]
    cmds = make_common("python -u " + abs_path("benchmarks/python/benchmark.py"), js, model, devices)
    def ap(s):
        nonlocal cmds
        cmds.append(s.strip())
    ap(f'-m llama_7b')
    ap(f'--mode plugin ')
    ap(f'--max_batch_size {max_batch_size}')
    ap(f'--max_input_len {max_input_len} ')
    ap(f'--max_output_len {max_output_len} ')
    return cmds

def make_cpp(js, model: str, devices: Optional[List[str]]):
    cmds = make_common(abs_path('cpp/build/benchmarks/gptSessionBenchmark'), js, model, devices)
    def ap(s):
        nonlocal cmds
        cmds.append(s.strip())
    ap(f'-m llama')
    return cmds
def run_bench(model: str, cpp: bool, test: bool, modeldef, world, devices: Optional[List[str]]):
    with open(os.path.join(model, "config.json"), 'r') as fp:
        js = json.load(fp)
        if cpp:
            cmds = make_cpp(js, model, devices)
        else:
            cmds = make_python(js, model, devices)

        # for i, line in enumerate(cmds):
        #     if i == 0:
        #         print(line)
        #     else:
        #         print(f"    {line}")
        cmd=' '.join(cmds)
        print(cmd)
        if not test:
            ret = os.system(cmd)
            if ret != 0:
                print(f"[BENCHMARK] error: fail to run test for {modeldef} {world}")
def record_result(path: str, modeldef: list, world: list):
    r = modeldef + world
    with open(path, 'a') as fp:
        fp.write(' '.join(str(i) for i in r) + '\n')
def load_record(path):
    ret = []
    if path is not None and os.path.exists(path):
        with open(path, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                line = line.strip()
                part = [int(s) for s in line.split(' ')]
                ret.append(part)
    return ret
def has_recorded(rec, modeldef, world):
    r = modeldef + world
    for line in rec:
        if line == r:
            return True
    return False
        
def bench(cpp: bool = False,
          test: bool = False,
          keep_tmpfile: bool = True,
          build_only: bool = False,
          devices: str = None,
          record: str = None,
          **kwargs):
    global g_worlds
    global g_moddefs
    tmpfiles = set()
    dst = kwargs['dst']
    if devices is not None:
        devices = devices.split(',')
    rec = load_record(record)
    for modeldef, world in product(g_moddefs, g_worlds):
        if has_recorded(rec, modeldef, world):
            continue
        if record is not None:
            record_result(record, modeldef, world)
        print(f'>>> start benchmark on bio {modeldef[0]} {modeldef[1]} {modeldef[2]} tp {world[0]} pp {world[1]}')
        model, tmppath = build_model(modeldef[0], modeldef[1], modeldef[2], world[0], world[1], **kwargs)
        print(f'model {model}')
        print(f'tmppath {tmppath}')
        if model is None:
            print(f"[BENCHMARK] error: fail to build model for {modeldef} {world}")
            continue
        if tmppath is not None:
            tmpfiles.add(tmppath)
        if not build_only:
            run_bench(model, cpp, test, modeldef, world, devices)
        # remove current engine, sometimes generated dir located in a subdirectory of dst,
        # like $dst/abc/model/*, we should remove abc, not model
        path = model
        if path.startswith(dst):
            while not os.path.samefile(os.path.dirname(path), dst):
                path = os.path.dirname(path)
        shutil.rmtree(path)
    if not keep_tmpfile:
        for path in tmpfiles:
            # remove all temp files generated for building engine
            shutil.rmtree(path)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpp', action='store_true', default=False, help="running cpp benchmark, or running python benchmark as default")
    parser.add_argument('--test', action='store_true', default=False, help="enable test mode")
    parser.add_argument('--keep_tmpfile', action='store_true', default=False, help="do not remove tmp files generated for model building")
    parser.add_argument('--build_only', action='store_true', default=False, help="skip test and build model only")
    # add argument defined by build.py, bio and parallel is decided by myself internally
    add_arguments(parser, excepts=['bio', 'parallel'])
    parser.add_argument('--devices', type=str, required=False, default=None, help='specify cuda devices used for test, format example: 1,2,3,4')
    parser.add_argument('--record', type=str, required=False, default=None, help='record history of test and skip those been tested, MUST be deleted when a new test starts')
    return parser.parse_args()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    args = parse_arguments()
    bench(**vars(args))