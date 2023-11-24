import os
import copy
from typing import List, Dict
# merge 2 performance results formated as markdown

pathpy = '/Users/gqjiang/logs/python.md'
pathcpp = '/Users/gqjiang/logs/cpp.md'


keys = ["max batch", "max input", "max output", "batch", "input", "output", 'world']
flts = ["latency", "token-per-sec", "gpu"]
cmpkey='cpp/py'
overkey = "over-world-1"
params = ["latency", "token-per-sec", overkey]
cppparams = ['cpp-'+p for p in params]
pyparams = ['py-'+p for p in params]
# md table hdr
hdrs = keys + ["gpu"] + cppparams + pyparams + [cmpkey]


def sort_key(d: Dict, ks):
    ret = 0
    for name in ks:
        ret = 10000 * ret + d[name]
    return ret

def split(line):
    segs = [s.strip() for s in line.split('|')]
    segs = [s for s in segs if len(s) > 0]
    return segs

def read(path):
    ret = []
    with open(path, 'r') as fp:
        hdrs = []
        idx = 0
        for line in fp.readlines():
            line = line.strip()
            idx += 1
            segs = split(line)
            if idx == 1:
                hdrs = segs
                for k in keys:
                    assert k in segs, f'k={k}, line: {idx} {line}'
                continue
            elif idx == 2:
                continue
            assert(len(segs) == len(hdrs))
            
            t = {}
            for k, v in zip(hdrs, segs):
                if v != '-':
                    if k in keys:
                        v = int(v)
                    elif k in flts:
                        v = float(v)
                t[k] = v
            if t['fail'] == '-':
                t['key'] = sort_key(t, keys)
                t['pkey'] = sort_key(t, keys[:-1])
                ret.append(t)
    return ret

def sorts(tab: List):
    tab.sort(key=lambda d: d['key'])
    return tab
    
def prefix(tab: List[Dict], s: str):
    for t in tab:
        for k in [p for p in params if p in t]:
            t[s+'-'+k] = t[k]
            del t[k]
def prt(tab: List):
    global keys
    for d in tab:
        print(','.join([str(d[k]) for k in keys]))

def world_cmp(tab):
    base = None
    for d in tab:

        if d['world'] == 1:
            base = d
        elif base is not None:
            if base is not None and d['pkey'] != base['pkey']:
                base = None
            else:
                d[overkey] = f"{d['token-per-sec'] / base['token-per-sec']:.3f}"
py= sorts(read(pathpy))
world_cmp(py)
prefix(py, 'py')

cpp= sorts(read(pathcpp))
world_cmp(cpp)
prefix(cpp, 'cpp')
    


# merge cpp to py
for d in cpp:
    target = None
    k = d['key']
    for pd in py:
        if pd['key'] == k:
            target = pd
            break
    if target is None:
        # no world found in py, append
        py.append(copy.deepcopy(d))
    else:
        # merge to target
        for k in cppparams:
            if k in d:
                target[k] = d[k]
        target[cmpkey] = f'{target["cpp-token-per-sec"]/target["py-token-per-sec"]:.3f}'

sorts(py)
output="/Users/gqjiang/logs/pycpp.md"
desc="""
Table Description:

| key | desc |
| --- | --- |
| max batch | max batch size of built model |
| max input	| max input length of built model |
| max output	| max output length of built model |
| batch	| benchmark task batch size |
| input	| benchmark task input length |
| output	| benchmark task output length |
| world	| model world size, 1: No TP & PP, 2: TP 2, 4: TP 2 PP 2 |
| gpu	| GPU peak memory(gb) |
| cpp-latency	| cpp benchmark average latency |
| cpp-token-per-sec	| cpp tokens per second |
| cpp-over-world-1	| cpp (token-per-sec / token-per-sec of world 1) |
| py-latency	| python benchmark average latency |
| py-token-per-sec	| python tokens per second |
| py-over-world-1	| python (token-per-sec / token-per-sec of world 1) |
| cpp/py | cpp token-per-sec / python token-per-sec |

Benchmark Data of llama-7b-hf:

"""
with open(output, 'w') as fp:
    fp.write(desc)
    fp.write('| ' + ' | '.join(hdrs) + ' |\n')
    fp.write("|" + " ---- |" * len(hdrs) + "\n")
    for d in py:
        strs = [str(d.get(k, '-')) for k in hdrs]
        fp.write('| ' + ' | '.join(strs) + ' |\n')
                


        
    
    

            







                

