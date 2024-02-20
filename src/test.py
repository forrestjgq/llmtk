
import json
import os
import pathlib
import sys


def load(file):
    with open(file, 'r') as fp:
        return json.load(fp)

def get_keys(js):
    return list(js.keys())

def get_and_cmp_keys(js1, js2):
    keys1 = get_keys(js1)
    keys2 = get_keys(js2)
    assert len(keys1) == len(keys2)
    assert all(k in keys2 for k in keys1)
    return keys1

def cmp(file1, file2):
    file1, file2 = pathlib.Path(file1), pathlib.Path(file2)
    js1, js2 = load(file1), load(file2)
    keys = get_and_cmp_keys(js1, js2)
    for k in keys:
        os.system('clear')
        print(f'{k}:')
        print(f'Input: {js1[k]["Input"]}:')
        print("=======================================================")
        print(f'{file1.name} {js1[k]["Tokens"]}:\n\t{js1[k]["Output"]}')
        print("=======================================================")
        print(f'{file2.name} {js2[k]["Tokens"]}:\n\t{js2[k]["Output"]}')
        print("=======================================================")
        input("Set a score(1~5):")

if __name__ == '__main__':
    assert len(sys.argv) == 3
    cmp(sys.argv[1], sys.argv[2])
