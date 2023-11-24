from __future__ import annotations
import os
import sys
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Dict, Optional, Set, Tuple
import copy


# This file parse tensorRT-LLM python benchmark logs and get markdown result

patterns = {
    "kv-header": "^ *>>>> max (.*)",
    "err-benchmark": "^\[BENCHMARK\].*error:.*",
    "kv-benchmark": "^\[BENCHMARK\] (.*)",
    "ls-skip": ".*skipping \((.*)\)",
    "str-RuntimeError": "^RuntimeError: (.*)",
    "err-RuntimeError": "TensorRT-LLM.ERROR.",
    "mode-py": "^CUDA_VISIBLE_DEVICES=.*benchmark.py.*",
    "mode-cpp": "^CUDA_VISIBLE_DEVICES=.*gptSessionBenchmark.*",
}
strkeys=['model_name', 'precision', 'compute_cap']
perfkeys={"tokens_per_sec": "tpu", "gpu_peak_mem(gb)": "gpumem"}

def splits(s: str):
    r = {}
    s = s.strip()
    g=s.split(' ')
    assert len(g) % 2 == 0, f'len(s) = {len(g)} str: {s}'
    sz = len(g) // 2
    for i in range(sz):
        k = g[i * 2]
        v = g[i*2+1]
        r[k] = v
    return r

def parse_line(line: str):
    global patterns
    for k, v in patterns.items():
        m = re.match(v, line)
        if m is not None:
            d=k.split('-')
            tp = d[0]
            name = d[1]
            if tp == 'kv':
                return name, splits(m.group(1))
            if tp == 'ls':
                return name, [int(n.strip()) for n in m.group(1).split(',')]
            if tp == 'str':
                return name, m.group(1)
            if tp == 'mode':
                return tp, name
            if tp == 'err':
                return name, None
            assert False, f'invalid key {k}'
    return None, None

@dataclass
class ParallelConfig():
    tp: int = 1
    pp: int = 1
    def __eq__(self, __value: object) -> bool:
        return object.tp == self.tp and object.pp == self.pp
    def __hash__(self) -> int:
        return self.tp * 10000 + self.pp
    def world(self):
        return self.tp * self.pp

@dataclass
class Test():
    batch: int = 0
    input: int = 0
    output: int = 0
    error: str = None
    results: dict = None
    def failed(self):
        return self.error is not None
    def tps(self):
        # for cpp only
        batch_size = int(self.results['batch_size'])
        outlen = int(self.results['output_length'])
        latency = float(self.results['latency(ms)'])
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        return tokens_per_sec
    def generate_tps(self):
        # for cpp only
        if "tokens_per_sec" not in self.results:
            self.results["tokens_per_sec"] = f'{self.tps():.2f}'
        
def sort_key(lst):
    ret = 0
    for v in lst:
        ret = 10000 * ret + v
    return ret

        
     
@dataclass
class Model(Test):
    worlds: Dict[ParallelConfig, List[Test]] = None # key: Parallel, value: list of Test, or error string
    def eq(self, b, i, o):
        return self.batch == b and self.input == i and self.output == o

    def same(self, other):
        return self.input == other.input and \
            self.output == other.output and \
            self.batch == other.batch

    
@dataclass
class PlainTest():
    """ class to maintain several tests with same config """
    max_batch: int = 0
    max_input: int = 0
    max_output: int = 0
    batch: int = 0
    input: int = 0
    output: int = 0
    parallel: ParallelConfig = None
    key: int = 0
    results: Dict[str, Dict[str, Any]] = None # key prefix, value: a dict of {key: segment name, value: segment value}(see Test.results)

    def get_key(self):
        if self.key == 0:
            vs = [self.max_batch, self.max_input, self.max_output, self.batch, self.input, self.output, self.parallel.tp, self.parallel.pp]
            self.key = sort_key(vs)
        return self.key
    def from_test(self, prefix: str, m: Model, parallel: ParallelConfig, t: Test, ks):
        self.max_batch = m.batch
        self.max_input = m.input
        self.max_output = m.output
        self.parallel = parallel
        self.batch = t.batch
        self.input = t.input
        self.output = t.output
        self.results = {
            prefix: {k: v for k, v in t.results.items() if k in ks and v != '-'}
        }
        return self.get_key()
    def add(self, other: PlainTest):
        for k, v in other.results.items():
            self.results[k] = v
    def merge_and_cmp(self, basePrefix: str, ks: Iterable[str]) -> List[Tuple[str, Any]]:
        """For each item in every dict of results, merge them into a list as tuple of (key, value) pair,
        the key here is prefixed with the corresponding key in results.

        For example, self.results['int8'] = {'latency': 1234.0, ...}, then a tuple of ('int8-latency', 1234.0) 
        should be appended to returing list.
        
        Further more, we are not just merging those items inside results, we should also compare them by dividing
        the value of those listed in `ks`. 
        
        Assuming ks contains 'latency', and also self.results['fp16'] = {'latency': 340.0, ...}, and `basePrefix`
        is 'fp16', then a new tuple ('latency (int8/fp16)', 1234.0/340.0) should be appended to the list

        Args:
            basePrefix (str): divide item value in self.results[*] by items in self.results[basePrefix] with same key
                              for comparing
            ks (Iterable[str]): define items to be compared

        Returns:
            List[Tuple[str, Any]] : pair of merged Tuple(prefixed-key, value) list
        """
        assert basePrefix in self.results
        merged: List[Tuple[str, Any]] = []
        base = self.results[basePrefix]
        for k, v in base.items():
            merged.append((basePrefix+'-'+k, v))
        for prefix, vs in self.results.items():
            if prefix != basePrefix:
                for k, v in vs.items():
                    merged.append((prefix+'-'+k, v))
                for k in ks:
                    if k in vs and k in base:
                        r = f'{float(vs[k])/float(base[k]):.02f}'
                        name = f'{k} ({prefix}/{basePrefix})'
                        merged.append((name, r))
        return merged

    
def parse_log(log: str, skip_err = False) -> List[Model]:
    """ parse python log file and return a collection of model tests"""
    mode = None # 'cpp', or 'py', or None
    models: List[Model] = []
    model: Optional[Model] = None
    curr_parallel: ParallelConfig = None
    composing: List[str] = []
    with open(log, 'r') as fp:
        nline = 0
        lines = fp.readlines()
        for line in lines:
            nline += 1
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                name, value = parse_line(line.strip())
                if name is None:
                    continue
                ishdr = name == 'header'
                # print(f'parsed: {nline} [{name}]: model is {model} hdr {ishdr}: {line}')
                if model is None and not ishdr:
                    continue
                if ishdr:
                    batch = int(value['batch']) 
                    input = int(value['in']) 
                    output = int(value['out'])
                    if 'world' in value:
                        w = int(value['world'])
                        if w == 1:
                            tp = 1
                            pp = 1
                        elif w == 2:
                            tp = 2
                            pp = 1
                        elif w == 4:
                            tp = 2
                            pp = 2
                        else:
                            assert False
                    else:
                        tp = int(value['tp'])
                        pp = int(value['pp'])
                    curr_parallel = ParallelConfig(tp=tp, pp=pp)
                    print(f'found model {batch} {input} {output} tp {curr_parallel.tp} pp {curr_parallel.pp}')
                    if model is None or not model.eq(batch, input, output):
                        model = Model(batch = int(value['batch']), input = int(value['in']),output = int(value['out']))
                        models.append(model)
                    if model.worlds is None:
                        model.worlds = {}
                    if curr_parallel not in model.worlds:
                        model.worlds[curr_parallel] = []
                elif name == 'mode':
                    print(f'enter mode {mode}, value {value}')
                    if mode is None:
                        mode = value
                    else:
                        assert mode == value
                    print(f'exit mode {mode}')
                elif name == 'benchmark':
                    if value is not None:
                        if mode is None:
                            if "model_name" in value:
                                mode = 'py'
                            else:
                                mode = 'cpp'
                        assert mode is not None
                        assert curr_parallel is not None and curr_parallel.world() == int(value.get('world_size', curr_parallel.world()))
                        def pint(k):
                            return int(value[k])
                        t = Test(batch=pint('batch_size'), input=pint('input_length'), output=pint('output_length'), results=value)
                        t.generate_tps()
                        model.worlds[curr_parallel].append(t)
                elif name == 'skip':
                    if not skip_err:
                        assert curr_parallel is not None
                        t = Test(batch=value[0], input=value[1], output=value[2], error="oom")
                        model.worlds[curr_parallel].append(t)
                elif name == "RuntimeError":
                    if not skip_err:
                        model.worlds[curr_parallel] = value
                else:
                    assert False, f'unknown line {name}'
            except Exception as e:
                print(f'line {nline} exception {e}')
                raise e
    return models
            
    

class Result:
    def __init__(self) -> None:
        self.mbatch = None
        self.minput = None
        self.moutput = None
        self.parallel: ParallelConfig = None
        self.fail = '-'
        self.batch = '-'
        self.input = '-'
        self.output = '-'
        self.latency = '-'
        self.token_per_sec = '-'
        self.gpu = '-'

    def set_world(self, b, i, o, parallel: ParallelConfig):
        self.mbatch = b
        self.minput = i
        self.moutput = o
        self.parallel = parallel
    def get_mbatch(self):
        return self.mbatch
    
# format
# | max batch | max input | max output | tp | pp | fail | batch | input | output | latency | token-per-sec | gpu |
def write_mdline(fp, r: Result):
    """write markdown line

    Args:
        fp (_type_): file
    """
    fp.write(f'| {r.mbatch} | {r.minput} | {r.moutput} | {r.parallel.tp} | {r.parallel.pp} | {r.fail} | {r.batch} | {r.input} | {r.output} | {r.latency} | {r.token_per_sec} | {r.gpu} |\n')

def write_md(models: List[Model], output_path: str):
    with open(output_path, 'w') as wp:
        wp.write('| max batch | max input | max output | tp | pp | fail | batch | input | output | latency | token-per-sec | gpu-peak-mem(gb) |\n')
        wp.write('| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |\n')
        for m in models:
            print(f'Model {m.batch} {m.input} {m.output}')
            if m.failed():
                print(f'\t{m.error}')
                continue
            for parallel, ts in m.worlds.items():
                r = Result()
                r.set_world(m.batch, m.input, m.output, parallel)
                print(f'\ttp size {parallel.tp} pp size {parallel.pp}')

                if isinstance(ts, str):
                    print(f'\t\tfailed: {ts}')
                    r.fail = 'world'
                    write_mdline(wp, r)
                    continue

                for t in ts:
                    rt = copy.deepcopy(r)
                    rt.batch, rt.input, rt.output = t.batch, t.input, t.output
                    
                    if t.failed():
                        print(f'\t\tfailed: {t.error}')
                        rt.fail = 'test'
                    else:
                        print(f'\t\t{t.batch} {t.input} {t.output}: tps {t.results["tokens_per_sec"]}')
                        rt.latency, rt.token_per_sec = t.results["latency(ms)"], t.results["tokens_per_sec"],
                        if "gpu_peak_mem(gb)" in t.results:
                            rt.gpu = t.results["gpu_peak_mem(gb)"]
                    write_mdline(wp, rt)
                
            
def merge_models(models: Dict[str, List[Model]], base: str, output_path: str):
    for k, v in models.items():
        print(f'{k}: {len(v)}')
    tests: Dict[int, PlainTest] = {}
    for prefix, ms in models.items():
        for m in ms:
            if m.failed():
                continue
            for parallel, ts in m.worlds.items():
                for t in ts:
                    if t.failed():
                        continue
                    pt = PlainTest()
                    k = pt.from_test(prefix, m, parallel, t, perfkeys.keys())
                    if k in tests:
                        tests[k].add(pt)
                    else:
                        tests[k] = pt
    tests = {k:v for k, v in tests.items() if len(v.results) == len(models)}

    merged: Dict[int, List[Tuple]] = {}
    for key, t in tests.items():
        merged[key] = t.merge_and_cmp(base, perfkeys.keys())

    keys = sorted(list(tests.keys()))
    hdrs = []
    for k in keys:
        t = tests[k]
        r = merged[k]
        strrs = ' '.join([f'{tp[0]}({tp[1]})' for tp in r])
        print(f'max bio {t.max_batch} {t.max_input} {t.max_output} tp {t.parallel.tp} pp {t.parallel.pp} bio {t.batch} {t.input} {t.output} {strrs}')
        if len(hdrs) == 0:
            hdrs = [
                "max batch",
                "max input",
                "max output",
                "batch",
                "input",
                "output",
                "tp",
                "pp",
            ] + [tp[0] for tp in r]
    hdrs = [s.replace('_', '-') for s in hdrs]
    def writeline(f, lst):
        f.write('| ' + ' | '.join([str(item) for item in lst]) + ' |\n')
    with open(output_path, "w") as fp:
        writeline(fp, hdrs)
        writeline(fp, ['----' for _ in range(len(hdrs))])
        for k in keys:
            t = tests[k]
            r = merged[k]
            lst = [t.max_batch, t.max_input, t.max_output, t.batch, t.input, t.output, t.parallel.tp, t.parallel.pp] + [tp[1] for tp in r]
            writeline(fp, lst)
    
    
        
            
            
        

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f'{sys.argv[0]} <parse|merge> <output> <log> [another-logs...]')
        sys.exit(2)
    action = sys.argv[1]
    output_file = sys.argv[2]
    if action == 'parse':
        # like: python bench_log.py parse int8/int8-py.md int8/int8-py.log
        # parse a log file generated by python performance test and write to markdown
        log_file = sys.argv[3]
        models = parse_log(log_file)
        write_md(models, output_file)
    elif action == 'merge':
        # like: python bench_log.py merge int8-to-py.md fp16:fp16/capture-all.txt int8:int8/int8-py.log 
        # merge several(here 2) logs which is generated by python performance test and
        # 1. merge those results
        # 2. compare the performance and gpu memory
        # output to a markdown
        assert len(sys.argv) >= 5
        logs = copy.deepcopy(sys.argv[3:])
        prefixes = [f't{i+1}' for i in range(len(logs))]
        params={}
        base = None
        for i, path in enumerate(logs):
            seps = path.split(':')
            if len(seps) == 2:
                prefixes[i] = seps[0]
                logs[i] = seps[1]
            params[prefixes[i]] = parse_log(logs[i], skip_err=True)
            if i == 0:
                base = prefixes[i]
        merge_models(params, base, output_file)
            

        
    