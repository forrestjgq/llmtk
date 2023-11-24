
import os
import re
from dataclasses import dataclass
from typing import List, Dict
import copy


# This file parse tensorRT-LLM cpp benchmark logs and get markdown result

patterns = {
    "kv-header": "^ *>>>> max (.*)",
    "kv-benchmark": "^\[BENCHMARK\] (.*)",
    "err-RuntimeError": "TensorRT-LLM.ERROR."
}

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
            if tp == 'err':
                return name, None
            assert False, f'invalid key {k}'
    return None, None

logs=["/Users/gqjiang/logs/cpp.log", "/Users/gqjiang/logs/cpp1.log"]

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
        batch_size = int(self.results['batch_size'])
        outlen = int(self.results['output_length'])
        latency = float(self.results['latency(ms)'])
        tokens_per_sec = round(batch_size * outlen / (latency / 1000), 2)
        return tokens_per_sec
    
@dataclass
class Model(Test):
    worlds: Dict[int, List[Test]] = None # key: world, value: list of Test, or error string
    def eq(self, b, i, o):
        return self.batch == b and self.input == i and self.output == o

    def same(self, other):
        return self.input == other.input and \
            self.output == other.output and \
            self.batch == other.batch

    
    
models: List[Model] = []
def find_model(b, i, o):
    for m in models:
        if m.eq(b, i, o):
            return m
    m = Model(batch=b, input=i, output=o)
    models.append(m)
    return m

for log in logs:
    model = None
    curr_world = None
    with open(log, 'r') as fp:
        nline = 0
        lines = fp.readlines()
        for line in lines:
            if line.endswith('\n'):
                line = line[:-1]
            print(f'line: {line}')
            nline += 1
                
            line = line.strip()
            if len(line) == 0:
                continue
            try:
                name, value = parse_line(line.strip())
                print(f'name: {name}')
                if name is None:
                    continue
                ishdr = name == 'header'
                if model is None and not ishdr:
                    continue
                if ishdr:
                    batch = int(value['batch']) 
                    input = int(value['in']) 
                    output = int(value['out'])
                    w = int(value['world'])
                    curr_world = w
                    print(f'found model {batch} {input} {output} world {w}')
                    model = find_model(batch, input, output)
                    if model.worlds is None:
                        model.worlds = {}
                    if w not in model.worlds:
                        model.worlds[w] = []
                elif name == 'benchmark':
                    w = curr_world
                    def pint(k):
                        return int(value[k])
                    t = Test(batch=pint('batch_size'), input=pint('input_length'), output=pint('output_length'), results=value)
                    model.worlds[w].append(t)
                elif name == "RuntimeError":
                    # model.worlds[curr_world] = "Error"
                    pass
                else:
                    assert False, f'unknown line {name}'
            except Exception as e:
                print(f'line {nline} exception {e}')
                raise e
                
        
# format
# | max batch | max input | max output | world | fail | batch | input | output | latency | token-per-sec | gpu |
output_file = '/Users/gqjiang/logs/cpp.md'

class Result:
    def __init__(self) -> None:
        self.mbatch = None
        self.minput = None
        self.moutput = None
        self.mworld = None
        self.fail = '-'
        self.batch = '-'
        self.input = '-'
        self.output = '-'
        self.latency = '-'
        self.token_per_sec = '-'

    def set_world(self, b, i, o, w):
        self.mbatch = b
        self.minput = i
        self.moutput = o
        self.mworld = w
    def get_mbatch(self):
        return self.mbatch
    
def write(fp, r: Result):
    fp.write(f'| {r.mbatch} | {r.minput} | {r.moutput} | {r.mworld} | {r.fail} | {r.batch} | {r.input} | {r.output} | {r.latency} | {r.token_per_sec} |\n')

with open(output_file, 'w') as wp:
    wp.write('| max batch | max input | max output | world | fail | batch | input | output | latency | token-per-sec |\n')
    wp.write('| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |\n')
    for m in models:
        print(f'Model {m.batch} {m.input} {m.output}')
        if m.failed():
            print(f'\t{m.error}')
        else:
            for w, ts in m.worlds.items():
                r = Result()
                r.set_world(m.batch, m.input, m.output, w)
                print(f'\tworld size {w}')
                if isinstance(ts, str):
                    print(f'\t\tfailed: {ts}')
                    r.fail = 'world'
                    write(wp, r)
                else:
                    for t in ts:
                        rt = copy.deepcopy(r)
                        rt.batch, rt.input, rt.output = t.batch, t.input, t.output
                        
                        if t.failed():
                            print(f'\t\tfailed: {t.error}')
                            rt.fail = 'test'
                        else:
                            print(f'\t\t{t.batch} {t.input} {t.output}: latency {t.results["latency(ms)"]} tps {t.tps()} ')
                            rt.latency, rt.token_per_sec = t.results["latency(ms)"], t.tps()
                        write(wp, rt)
            
        
        

    
