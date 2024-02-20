import argparse
from ctypes import ArgumentError
from dataclasses import dataclass
import json
import os
import shutil
from datetime import datetime

import torch

from pathlib import Path


def is_llama(model_type):
    return model_type in ["llama", "mistral", "llava"]


def is_baichuan(model_type):
    return model_type == "baichuan"


def is_falcon(model_type):
    return model_type == "falcon"


def is_chatglm(model_type):
    return model_type == "chatglm"


@dataclass
class Config:
    model_name: str = None
    qt: str = None
    batch: int = 0
    input: int = 0
    output: int = 0
    tp: int = 1
    pp: int = 1
    src: str = None
    dst: str = None
    dtype: str = "float16"
    model_type: str = None

    def __post_init__(self):
        with open(os.path.join(self.src, "config.json"), "r") as fp:
            js = json.load(fp)
            self.model_type = js["model_type"].lower()
            attention_heads = js["num_attention_heads"]
            assert (
                attention_heads % self.tp == 0
            ), f"num attention heads {attention_heads} does not support tensor parallel {self.tp}"

    def _get_model_type(self):
        with open(os.path.join(self.src, "config.json"), "r") as fp:
            return json.load(fp)["model_type"].lower()

    def world(self):
        return self.tp * self.pp

    def dst_path(self):
        return self.dst


class Options:
    def __init__(self, trtllm: str, devices) -> None:
        self.d = {}
        self.trtllm = trtllm
        self.devices = devices

    @staticmethod
    def _key(key):
        key = key.strip()
        if key.startswith("-"):
            prefix = ""
        elif len(key) == 1:
            prefix = "-"
        else:
            prefix = "--"
        return prefix + key

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

    def generate(self, file, disabled=None):
        py = self._path(file)
        s = []
        for k, v in self.d.items():
            if disabled is None or k not in disabled:
                s.append(k)
                if v is not None:
                    s.append(str(v))
        prefix = ""
        if self.devices and len(self.devices) > 0:
            prefix = f"CUDA_VISIBLE_DEVICES={self.devices} "
        return prefix + f"python -u {py} " + " ".join(s)


class Exec:
    def __init__(self, trtllm, devices, model_type) -> None:
        self.trtllm = trtllm
        self.model_type = model_type
        self.extra_options = {}
        self.disabled_options = set()
        self.cmds = []
        if is_llama(self.model_type):
            self.hf_cvt_file = "examples/llama/hf_llama_convert.py"
            self.quant_file = "examples/llama/quantize.py"
            self.build_file = "examples/llama/build.py"
        elif is_baichuan(self.model_type):
            self.hf_cvt_file = "examples/baichuan/hf_baichuan_convert.py"
            self.quant_file = None
            self.build_file = "examples/baichuan/build.py"
        elif is_chatglm(self.model_type):
            self.hf_cvt_file = None
            self.quant_file = "examples/chatglm/quantize.py"
            self.build_file = "examples/chatglm/build.py"
        elif is_falcon(self.model_type):
            self.hf_cvt_file = None
            self.quant_file = "examples/falcon/quantize.py"
            self.build_file = "examples/falcon/build.py"
        else:
            raise Exception("no impl found for model type " + self.model_type)

        self.devices = devices
        

    def remove_option(self, *keys):
        for key in keys:
            if len(key) == 1:
                key = "-" + key
            else:
                key = "--" + key
            self.disabled_options.add(key)

    def add_option(self, key: str, value: any):
        self.extra_options[key] = value
        return self

    def _new_option(self, clean=False):
        opts = Options(self.trtllm, self.devices)
        if not clean:
            for k, v in self.extra_options.items():
                opts.add(k, v)
        return opts

    def make_fp16(self, cfg):
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        return self._build(opt, cfg), None

    def _make_fp8_qt(self, src, dst, calib_size=512):
        opt = self._new_option(clean=True)
        cmd = (
            opt.add("model_dir", src)
            .add("dtype", "float16")
            .add("qformat", "fp8")
            .add("export_path", dst)
            .add("calib_size", calib_size)
            .generate(self.quant_file)
        )
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
        # assert cfg.pp == 1
        sqdst = os.path.join(dst, f"{cfg.tp}-gpu")
        if os.path.exists(sqdst):
            return sqdst
        opt = self._new_option(clean=True)
        cmd = (
            opt.add("i", cfg.src)
            .add("o", dst)
            .add("-sq", sq)
            .add("tensor-parallelism", cfg.tp)
            .add("storage-type", "fp16")
            .generate(self.hf_cvt_file)
        )
        ret = self._run_cmd(cmd)
        if ret == 0:
            return sqdst
        
        # remove tree to avoid reuse failed
        shutil.rmtree(sqdst)
        return None

    def make_sq(self, cvtdst, cfg: Config, sq=0.8):
        sqdst = self._cvt_sq(cvtdst, cfg, sq=sq)
        if sqdst is None:
            return None, None
        opt = self._new_option()
        self._make_cfg_options(opt, cfg)
        self._make_common_options(opt)
        opt.remove("model_dir")
        opt.add("bin_model_dir", sqdst)
        opt.add("use_smooth_quant")
        opt.add("per_token")
        opt.add("per_channel")
        return self._build(opt, cfg), sqdst

    def _cvt_int8_kv(self, dst, cfg: Config):
        kvdst = os.path.join(dst, f"{cfg.tp}-gpu")
        if os.path.exists(kvdst):
            shutil.rmtree(kvdst)
            # print(f"quant model exists: {kvdst}")
            # return kvdst
        opt = self._new_option(clean=True)
        cmd = (
            opt.add("i", cfg.src)
            .add("o", dst)
            .add("t", "fp16")
            .add("calibrate-kv-cache")
            .add("tensor-parallelism", cfg.tp)
            .generate(self.hf_cvt_file)
        )
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
            shutil.rmtree(dst)
            # return dst
        opt = self._new_option(clean=True)
        cmd = (
            opt.add("model_dir", src)
            .add("dtype", "float16")
            .add("qformat", "int4_awq")
            .add("export_path", dst)
            .add("calib_size", calib_size)
            .generate(self.quant_file)
        )
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

    def _post_build(self, cfg: Config):
        src = Path(cfg.src)
        dst = Path(cfg.dst_path()) / "model"
        dst.mkdir(exist_ok=True)
        for f in src.iterdir():
            if not f.is_file():
                continue
            name = f.name.lower()
            if name.endswith("json") or "token" in name:
                shutil.copy(f, dst)
        if self.model_type == "llava":
            vt = src / "vision_tower"
            assert vt.exists()
            vtdst = dst / "vision_tower"
            if vtdst.exists():
                shutil.rmtree(vtdst)
            shutil.copytree(vt, vtdst)
            with open(src / "pytorch_model.bin.index.json", "r") as fp:
                js = json.load(fp)
                wmap = js["weight_map"]
                bins = {
                    v for k, v in wmap.items() if k.startswith("model.mm_projector")
                }
                assert len(bins) == 1
                w = torch.load(src / bins.pop())
                prefix = "model.mm_projector."
                w = {k[len(prefix) :]: v for k, v in w.items() if k.startswith(prefix)}
                for k in w.keys():
                    print("\t", k)
                torch.save(w, dst / "mm_projector.bin")
        return cfg.dst_path()

    def _build(self, opt: Options, cfg):
        ret = self._run_cmd(
            opt.generate(self.build_file, disabled=self.disabled_options)
        )
        if ret == 0:
            return self._post_build(cfg)
        return None

    @staticmethod
    def _make_common_options(opt: Options):
        opt.add("remove_input_padding", None)
        opt.add("use_gpt_attention_plugin", "float16")
        opt.add("use_gemm_plugin", "float16")
        opt.add("enable_context_fmha", None)
        # failed while loading embedding weight if tp > 1
        # and baichuan does not support
        # opt.add("use_parallel_embedding")
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
        opt.add("max_num_tokens", cfg.batch * cfg.input)

    def _run_cmd(self, cmd):
        self.cmds.append(cmd)
        print(f">> {cmd}")
        ret = os.system(cmd)
        if ret != 0:
            print(f"cmd execute failed: {ret}")
        return ret

    def _run(self):
        cmd = self._make_must_options()._make_common_options()._generate()
        print(f">> {cmd}")
        ret = os.system(cmd)
        if ret == 0:
            print("engine built to: " + self.cfg.dst)
        else:
            print("engine build failed")


def _get_chatglm_model_name(cfg: Config):
    model_name = cfg.model_name
    if model_name is None or len(cfg.model_name) == 0:
        model_name = Path(cfg.src).parts[-1]
    segs = set(model_name.replace("_", "-").lower().split("-"))
    p1 = {"chatglm3", "chatglm2", "chatglm", "glm"}
    sec = list(p1.intersection(segs))
    assert len(sec) == 1, "unknown chatglm model " + model_name
    segs.remove(sec[0])
    realname = sec[0]

    p1 = {"6b", "10b"}
    sec = list(p1.intersection(segs))
    assert len(sec) == 1, "unknown chatglm model " + model_name
    realname = realname + "_" + sec[0]
    segs.remove(sec[0])

    p1 = {"32k", "base"}
    sec = list(p1.intersection(segs))
    if len(sec) == 1:
        realname = realname + "_" + sec[0]

    names = [
        "chatglm_6b",
        "chatglm2_6b",
        "chatglm2_6b_32k",
        "chatglm3_6b",
        "chatglm3_6b_base",
        "chatglm3_6b_32k",
        "glm_10b",
    ]
    assert realname in names, f"unknown predicted model name {realname}"
    return realname


def build_falcon(cfg: Config, exec: Exec, tmpdir: str = None):
    qt = cfg.qt
    if qt is None:
        output_path, tmp_path = exec.make_fp16(cfg)
    else:
        raise Exception("not impl")
        if qt == "sq":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_sq(tmpdir, cfg, sq=0.8)
        elif qt == "int8kv":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_w8kv8(tmpdir, cfg)
        elif qt == "fp8":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_fp8(tmpdir, cfg)
        elif qt == "awq":
            assert tmpdir is not None
            cvtdst = os.path.join(tmpdir, "llama-7b-4bit-gs128-awq.pt")  # todo
            output_path, tmp_path = exec.make_awq(cvtdst, cfg)
    return output_path, tmp_path


def build_chatglm(cfg: Config, exec: Exec, tmpdir: str = None):
    assert cfg.pp == 1, "chatglm does not support pipeline parallel"
    model_name = _get_chatglm_model_name(cfg)
    exec.add_option("model_name", model_name)

    qt = cfg.qt
    if qt is None:
        output_path, tmp_path = exec.make_fp16(cfg)
    else:
        raise Exception("not impl")
        if qt == "sq":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_sq(tmpdir, cfg, sq=0.8)
        elif qt == "int8kv":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_w8kv8(tmpdir, cfg)
        elif qt == "fp8":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_fp8(tmpdir, cfg)
        elif qt == "awq":
            assert tmpdir is not None
            cvtdst = os.path.join(tmpdir, "llama-7b-4bit-gs128-awq.pt")  # todo
            output_path, tmp_path = exec.make_awq(cvtdst, cfg)
    return output_path, tmp_path


def build_llama(cfg: Config, exec: Exec, tmpdir: str = None):
    exec.add_option("use_lookup_plugin", None)
    if cfg.model_type == "mistral":
        # load config.json in model src dir and get max_position_embeddings to
        # set option max_input_len
        # see readme in trtllm/example/llama/
        with open(os.path.join(cfg.src, "config.json"), "r") as fp:
            js = json.load(fp)
            exec.add_option("max_input_len", js["max_position_embeddings"])
    elif cfg.model_type == "llava":
        if cfg.input < 576:
            raise Exception(
                "Llava require 576 ids for each image feature, so the input size should large than that"
            )
        # current trtllm has full support for llava, but using CPU to build engine
        # which is extreamly slow.
        #
        # so we still treat llava as llama now unless trtllm using GPU to build
        exec.add_option("model_type", "llama")

    qt = cfg.qt
    if qt is None:
        output_path, tmp_path = exec.make_fp16(cfg)
    else:
        if tmpdir is None:
            tmpdir = cfg.dst
        if qt.startswith("sq"):
            qts = qt.split('=')
            if len(qts) == 1:
                param = 0.8
            elif len(qts) == 2:
                param = float(qts[1])
            else:
                raise ArgumentError(qt)
            output_path, tmp_path = exec.make_sq(tmpdir, cfg, sq=param)
        elif qt == "int8kv":
            output_path, tmp_path = exec.make_w8kv8(tmpdir, cfg)
        elif qt == "fp8":
            output_path, tmp_path = exec.make_fp8(tmpdir, cfg)
        elif qt == "awq":
            cvtdst = os.path.join(tmpdir, "llama-7b-4bit-gs128-awq.pt")  # todo
            output_path, tmp_path = exec.make_awq(cvtdst, cfg)
    return output_path, tmp_path


def build_baichuan(cfg: Config, exec: Exec, tmpdir: str = None):
    assert cfg.pp == 1, "baichuan does not support pipeline parallel"
    model_name = cfg.model_name
    if model_name is None or len(model_name) == 0:
        model_name = Path(cfg.src).parts[-1]  # try to get from file path
    segs = model_name.lower().split("-")
    if "baichuan2" in segs:
        ver = "v2"
    elif "baichuan" in segs:
        ver = "v1"
    else:
        raise Exception(
            f"unknown model name {model_name}, we accept name like Baichuan2-7B-Chat"
        )
    if "7b" in segs:
        ver = ver + "_7b"
    elif "13b" in segs:
        ver = ver + "_13b"
    else:
        raise Exception(
            f"unknown model name {model_name}, we accept name like Baichuan2-7B-Chat"
        )
    exec.add_option("model_version", ver)
    exec.remove_option("tp_size", "pp_size")

    qt = cfg.qt
    if qt is None:
        output_path, tmp_path = exec.make_fp16(cfg)
    else:
        # todo
        raise Exception("not impl")
        if qt == "sq":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_sq(tmpdir, cfg, sq=0.8)
        elif qt == "int8kv":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_w8kv8(tmpdir, cfg)
        elif qt == "fp8":
            assert tmpdir is not None
            output_path, tmp_path = exec.make_fp8(tmpdir, cfg)
        elif qt == "awq":
            assert tmpdir is not None
            cvtdst = os.path.join(tmpdir, "llama-7b-4bit-gs128-awq.pt")  # todo
            output_path, tmp_path = exec.make_awq(cvtdst, cfg)
    return output_path, tmp_path

def get_engine_cfg(engine):
    path = Path(engine)
    found_engine = False
    cfg = None
    for p in path.iterdir():
        if p.suffix.lower() == ".engine":
            found_engine = True
        if p.parts[-1] == "config.json":
            cfg = str(p)
        if cfg is None and str(p).endswith("config.json"):
            cfg = str(p)
    if not found_engine:
        return None
    return cfg

def write_build_params(path, params):
    app=os.getenv('APP_IMAGE', "")
    llmtk=os.getenv('APP_LLMTK_COMMITID', '')
    params['app'] = app
    params['llmtk'] = llmtk
    params['date'] = str(datetime.now())
    with open(path, 'w') as fp:
        json.dump(params, fp)
def build(
    trtllm: str = None,
    name: str = None,
    bio: str = None,
    parallel: str = None,
    src: str = None,
    dst: str = None,
    qt: str = None,
    tmpdir: str = None,
    devices: str = None,
    keep_intermediate: bool = False,
    **kwargs,
):
    assert trtllm is not None
    assert bio is not None
    assert parallel is not None
    assert src is not None
    assert dst is not None

    v = [int(s) for s in bio.split(":")]
    batch, input, output = v[0], v[1], v[2]
    v = [int(s) for s in parallel.split(":")]
    tp, pp = v[0], v[1]
    cfg = Config(
        model_name=name,
        qt=qt,
        batch=batch,
        input=input,
        output=output,
        tp=tp,
        pp=pp,
        src=src,
        dst=dst,
    )
    model_type = cfg.model_type

    pdst = Path(dst)
    if not pdst.exists():
        pdst.mkdir(parents=True, exist_ok=True)

    exec = Exec(trtllm, devices, model_type)
    builders = [
        [is_llama, build_llama],
        [is_baichuan, build_baichuan],
        [is_chatglm, build_chatglm],
        [is_falcon, build_falcon],
    ]
    found = False
    for b in builders:
        if b[0](model_type):
            output_path, tmp_path = b[1](cfg, exec, tmpdir)
            found = True
            break
    if not found:
        raise Exception("unknown model type " + model_type)
    assert output_path is not None
    assert os.path.exists(output_path)

    
    filepath = Path(get_engine_cfg(output_path))
    assert filepath.exists(), f"engine config {filepath} not found"
    if filepath.parts[-1] != "config.json":
        # trtllm sucks on chatglm model building because they write chatglmxxx-config.json
        # instead of config.json, which can not be recognized by backend
        if "chatglm" in name.lower():
            shutil.copyfile(filepath, filepath.parent / "config.json")
        else:
            raise Exception("lack of impl")

    params = {
        "src": src,
        "dst": dst,
        "name": name,
        "tp_size": tp,
        "pp_size": pp,
        "batch": batch,
        "max_input_len": input,
        "max_output_len": output,
        "qt": qt,
        "cmds": exec.cmds
    }
    write_build_params(Path(output_path)/'build_params.json', params)

    if tmp_path is not None:
        assert output_path != tmp_path
        if not keep_intermediate:
            shutil.rmtree(tmp_path)
    return output_path, tmp_path


def add_arguments(parser: argparse.ArgumentParser, excepts=[]):
    if "trtllm" not in excepts:
        parser.add_argument(
            "--trtllm", type=str, default="/app/tensorrt_llm", help="TensorRT-LLM path"
        )
    if "name" not in excepts:
        # Baichuan require model version, which coming from model official name, to determine
        # how to build the engine, in which case if
        parser.add_argument(
            "--name",
            type=str,
            required=True,
            help="model official name, critial parameter impacting model running",
        )
    if "bio" not in excepts:
        parser.add_argument(
            "--bio", type=str, required=False, help="<batch>:<input len>:<output len>"
        )
    if "parallel" not in excepts:
        parser.add_argument(
            "--parallel",
            type=str,
            required=False,
            default="1:1",
            help="<tp size>:<pp size>",
        )
    if "src" not in excepts:
        parser.add_argument("--src", type=str, required=False, default=None)
    if "dst" not in excepts:
        parser.add_argument("--dst", type=str, required=False, default=None)
    if "devices" not in excepts:
        parser.add_argument(
            "--devices",
            type=str,
            default=None,
            help="specify cuda devices to use, like `0,1,2,3`, use any device available if not set",
        )
    if "qt" not in excepts:
        parser.add_argument(
            "--qt", type=str, default=None
        )
        # parser.add_argument(
        #     "--qt", type=str, default=None, choices=["sq", "int8kv", "fp8", "awq"]
        # )
    if "tmpdir" not in excepts:
        parser.add_argument("--tmpdir", type=str, default=None)
    if "keep_intermediate" not in excepts:
        # Baichuan require model version, which coming from model official name, to determine
        # how to build the engine, in which case if
        parser.add_argument(
            "--keep_intermediate",
            action="store_true",
            default=False,
            help="turn this on to keep intermediate files generated for building",
        )


def parse_arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    build(**vars(args))
