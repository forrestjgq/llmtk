import argparse
from dataclasses import dataclass, field
import json
import os
import sys
from typing import Dict, Optional

import torch

@dataclass
class Config:
    model: str = None
    qt: str = None
    batch: int = 0
    input: int = 0
    output: int = 0
    tp: int = 1
    pp: int = 1
    src: str = None
    dst: str = None
    prefix: str = None
    dtype: str = "float16"
    model_type: str = None
    direct_save: bool = False

    def __post_init__(self):
        self.model_type = self._get_model_type()
    
            
    def _get_model_type(self):
        with open(os.path.join(self.src, 'config.json'), 'r') as fp:
            return json.load(fp)['model_type'].lower()

    def world(self):
        return self.tp * self.pp

    def dst_path(self):
        if self.direct_save:
            return self.dst
        segs = [self.model, self.prefix, self.batch, self.input, self.output, self.tp, self.pp]
        segs = [str(p) for p in segs if p is not None]
        return os.path.join(self.dst, '_'.join(segs))

class Options:
    def __init__(self, trtllm: str, devices) -> None:
        self.d = {}
        self.trtllm = trtllm
        self.devices = devices
    @staticmethod
    def _key(key):
        key = key.strip()
        if len(key) == 1:
            prefix='-'
        else:
            prefix = '--'
        return prefix+key
        
    def add(self, key, value=None):
        key = self._key(key)
        self.d[key] = value
        return self
    def remove(self, key):
        key = self._key(key)
        if key in self.d:
            del self.d[key]

    def _path(self, file):
        return os.path.join(self.trtllm, file)

    def generate(self, file):
        py = self._path(file)
        s = []
        for k, v in self.d.items():
            s.append(k)
            if v is not None:
                s.append(str(v))
        prefix = ''
        if self.devices and len(self.devices) > 0:
            prefix = f"CUDA_VISIBLE_DEVICES={self.devices} "
        return prefix + f"python -u {py} " + " ".join(s)

class Exec:
    def __init__(self, trtllm, devices) -> None:
        self.trtllm = trtllm
        self.hf_cvt_file = "examples/llama/hf_llama_convert.py"
        self.quant_file = "examples/llama/quantize.py"
        self.build_file = "examples/llama/build.py"
        self.extra_options = {}
        self.devices = devices

    def add_option(self, key: str, value: any):
        self.extra_options[key] = value
        return self

    def _new_option(self):
        opts = Options(self.trtllm, self.devices)
        for k, v in self.extra_options.items():
            opts.add(k, v)
        return opts

    def make_fp16(self, cfg):
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        return self._build(opt, cfg), None

    def _make_fp8_qt(self, src, dst, calib_size=512):
        opt = self._new_option()
        cmd = opt.add("model_dir", src)\
                .add("dtype", "float16")\
                .add("qformat", "fp8")\
                .add("export_path", dst)\
                .add("calib_size", calib_size)\
                .generate(self.quant_file)
        ret = self._run_cmd(cmd)
        if ret == 0:
            return os.path.join(dst, "llama_tp1_rank0.npz")
        return None

    def make_fp8(self, qdst: str, cfg: Config):
        qtout = self._make_fp8_qt(cfg.src, qdst, cfg)
        if qtout is None:
            return None, None
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        opt.add("quantized_fp8_model_path", qtout)
        opt.add("enable_fp8")
        opt.add("fp8_kv_cache")
        return self._build(opt, cfg), qtout
    
    def _cvt_sq(self, dst, cfg: Config, sq=0.8):
        assert cfg.pp == 1
        sqdst = os.path.join(dst, f'{cfg.tp}-gpu')
        if os.path.exists(sqdst):
            return sqdst
        opt = self._new_option()
        cmd = opt.add("i", cfg.src).add("o", dst).add("-sq", sq).add("tensor-parallelism", cfg.tp).add("storage-type", "fp16")\
            .generate(self.hf_cvt_file)
        ret = self._run_cmd(cmd)
        if ret == 0:
            return sqdst
        return None
        
    def make_sq(self, cvtdst, cfg: Config, sq=0.8):
        sqdst = self._cvt_sq(cvtdst, cfg, sq=sq)
        if sqdst is None:
            return None, None
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        opt.remove("model_dir")
        opt.add("ft_model_dir", sqdst)
        opt.add("use_smooth_quant")
        opt.add("per_token")
        opt.add("per_channel")
        return self._build(opt, cfg), sqdst

    def _cvt_int8_kv(self, dst, cfg: Config):
        kvdst = os.path.join(dst, f'{cfg.tp}-gpu')
        if os.path.exists(kvdst):
            print(f"quant model exists: {kvdst}")
            return kvdst
        opt = self._new_option()
        cmd = opt.add("i", cfg.src)\
                .add("o", dst)\
                .add("t", "fp16")\
                .add("calibrate-kv-cache")\
                .add("tensor-parallelism", cfg.tp)\
                .generate(self.hf_cvt_file)
        ret = self._run_cmd(cmd)
        if ret == 0:
            return kvdst
        return None

    def make_w8kv8(self, cvtdst, cfg: Config):
        kvdst = self._cvt_int8_kv(cvtdst, cfg)
        if kvdst is None:
            return None, None
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        opt.remove("model_dir")
        opt.add("ft_model_dir", kvdst)
        opt.add("int8_kv_cache")
        opt.add("use_weight_only")
        return self._build(opt, cfg), kvdst

    def _cvt_awq(self, src, dst, calib_size=32):
        if os.path.exists(dst):
            return dst
        opt = self._new_option()
        cmd = opt.add("model_dir", src)\
                .add("dtype", "float16")\
                .add("qformat", "int4_awq")\
                .add("export_path", dst)\
                .add("calib_size", calib_size)\
                .generate(self.quant_file)
        ret = self._run_cmd(cmd)
        if ret == 0:
            return dst
        return None

    
    def make_awq(self, cvtdst, cfg: Config, calib_size=32):
        awq = self._cvt_awq(cfg.src, cvtdst, calib_size=calib_size)
        if awq is None:
            return None, None
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        opt.add("quant_ckpt_path", awq)
        opt.add("use_weight_only")
        opt.add("weight_only_precision", "int4_awq")
        opt.add("per_group")
        return self._build(opt, cfg), awq
        
        
    def _build(self, opt, cfg):
        ret = self._run_cmd(opt.generate(self.build_file))
        if ret == 0:
            return cfg.dst_path()
        return None
    @staticmethod
    def _make_common_options(opt: Options):
        opt.add("remove_input_padding", None)
        opt.add("use_gpt_attention_plugin", "float16")
        opt.add("use_gemm_plugin", "float16")
        opt.add("enable_context_fmha", None)
        # opt.add("use_parallel_embedding") # failed while loading embedding weight if tp > 1
        opt.add("use_inflight_batching")
        opt.add("paged_kv_cache")
    @staticmethod
    def _make_cfg_options(opt: Options, cfg: Config):
        opt.add("model_dir", cfg.src)
        opt.add("dtype", cfg.dtype)
        opt.add("output_dir", cfg.dst_path())
        opt.add("max_batch_size", cfg.batch)
        opt.add("max_input_len", cfg.input)
        opt.add("max_output_len", cfg.output)
        opt.add("world_size", cfg.world())
        opt.add("tp_size", cfg.tp)
        opt.add("pp_size", cfg.pp)
        
    @staticmethod
    def _run_cmd(cmd):
        print(f'>> {cmd}')
        ret = os.system(cmd)
        if ret != 0:
            print(f"cmd execute failed: {ret}")
        return ret

    def _run(self):
        cmd = self._make_must_options()._make_common_options()._generate()
        print(f'>> {cmd}')
        ret = os.system(cmd)
        if ret == 0:
            print("engine built to: " + self.cfg.dst)
        else:
            print("engine build failed")
        
def build_llama(cfg: Config, exec: Exec, tmpdir: str = None):
    name = cfg.model
    if cfg.model_type == 'mistral':
        # load config.json in model src dir and get max_position_embeddings to
        # set option max_input_len
        # see readme in trtllm/example/llama/
        with open(os.path.join(cfg.src, 'config.json'),'r') as fp:
            js = json.load(fp)
            exec.add_option("max_input_len", js['max_position_embeddings'])
    qt = cfg.qt
    if qt is None:
        cfg.prefix = "fp16"
        output_path, tmp_path = exec.make_fp16(cfg)
    else:
        if qt == "sq":
            cfg.prefix = "sq0.8"
            assert tmpdir is not None
            output_path, tmp_path = exec.make_sq(tmpdir, cfg, sq=0.8)
        elif qt == "int8kv":
            cfg.prefix = qt
            assert tmpdir is not None
            output_path, tmp_path = exec.make_w8kv8(tmpdir, cfg)
        elif qt == "fp8":
            cfg.prefix = qt
            assert tmpdir is not None
            output_path, tmp_path = exec.make_fp8(tmpdir, cfg)
        elif qt == "awq":
            cfg.prefix = qt
            assert tmpdir is not None
            cvtdst = os.path.join(tmpdir, "llama-7b-4bit-gs128-awq.pt")  # todo
            output_path, tmp_path = exec.make_awq(cvtdst, cfg)
    return output_path, tmp_path


def build(trtllm: str = None,
          name: str = None,
          bio: str = None,
          parallel: str = None,
          src: str = None,
          dst: str = None,
          qt: str = None,
          tmpdir: str=None,
          direct_save: bool = False,
          devices: str = None,
          **kwargs):
    assert trtllm is not None
    assert bio is not None
    assert parallel is not None
    assert src is not None
    assert dst is not None

    v = [int(s) for s in bio.split(":")]
    batch, input, output = v[0], v[1], v[2]
    v = [int(s) for s in parallel.split(":")]
    tp, pp = v[0], v[1]
    cfg = Config(model=name, qt=qt, batch=batch, input=input, output=output, tp=tp, pp=pp, src=src, dst=dst, direct_save=direct_save)
    exec = Exec(trtllm, devices)
    model_type = cfg.model_type
    if model_type == 'llama' or model_type == 'mistral':
        output_path, tmp_path = build_llama(cfg, exec, tmpdir)
    else:
        raise Exception("unknown model type " + model_type)
    assert output_path is not None
    assert os.path.exists(output_path)
    return output_path, tmp_path

            
    
        
def add_arguments(parser: argparse.ArgumentParser, excepts=[]):
    if 'trtllm' not in excepts:
        parser.add_argument('--trtllm', type=str, default=None, help="TensorRT-LLM path")
    if 'name' not in excepts:
        parser.add_argument('--name', type=str, required=False, help="model name, used to name the engine directory")
    if 'bio' not in excepts:
        parser.add_argument('--bio', type=str, required=False, help="<batch>:<input len>:<output len>")
    if 'parallel' not in excepts:
        parser.add_argument('--parallel', type=str, required=False, default='1:1', help="<tp size>:<pp size>")
    if 'src' not in excepts:
        parser.add_argument('--src', type=str, required=False, default=None)
    if 'dst' not in excepts:
        parser.add_argument('--dst', type=str, required=False, default=None)
    if 'direct-save' not in excepts:
        parser.add_argument('--direct-save', action='store_true', default=False)
    if 'devices' not in excepts:
        parser.add_argument('--devices', type=str, default=None, help='specify cuda devices to use, like `0,1,2,3`')
    if 'qt' not in excepts:
        parser.add_argument("--qt",
                            type=str,
                            default=None,
                            choices=["sq", "int8kv", "fp8", "awq"])
    if 'tmpdir' not in excepts:
        parser.add_argument('--tmpdir', type=str, default=None)
    

def parse_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    build(**vars(args))