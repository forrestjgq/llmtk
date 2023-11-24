# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import json
from pathlib import Path
from typing import List
import os

import numpy as np
import torch
from transformers import LlamaTokenizer

import tensorrt_llm
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import ModelConfig, SamplingConfig

from build import get_engine_name  # isort:skip

EOS_TOKEN = 2
PAD_TOKEN = 2


def throttle_generator(generator, stream_interval):
    for i, out in enumerate(generator):
        if not i % stream_interval:
            yield out

    if i % stream_interval:
        yield out


def read_config(config_path: Path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    use_gpt_attention_plugin = config['plugin_config']['gpt_attention_plugin']
    remove_input_padding = config['plugin_config']['remove_input_padding']
    dtype = config['builder_config']['precision']
    tp_size = config['builder_config']['tensor_parallel']
    pp_size = config['builder_config']['pipeline_parallel']
    batch_size = config['builder_config']['max_batch_size']
    input_len = config['builder_config']['max_input_len']
    output_len = config['builder_config']['max_output_len']
    world_size = tp_size * pp_size
    assert world_size == tensorrt_llm.mpi_world_size(), \
        f'Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})'
    num_heads = config['builder_config']['num_heads'] // tp_size
    hidden_size = config['builder_config']['hidden_size'] // tp_size
    vocab_size = config['builder_config']['vocab_size']
    num_layers = config['builder_config']['num_layers']
    num_kv_heads = config['builder_config'].get('num_kv_heads', num_heads)
    paged_kv_cache = config['plugin_config']['paged_kv_cache']
    tokens_per_block = config['plugin_config']['tokens_per_block']
    quant_mode = QuantMode(config['builder_config']['quant_mode'])
    if config['builder_config'].get('multi_query_mode', False):
        tensorrt_llm.logger.warning(
            "`multi_query_mode` config is deprecated. Please rebuild the engine."
        )
        num_kv_heads = 1
    num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
    use_custom_all_reduce = config['plugin_config'].get('use_custom_all_reduce',
                                                        False)

    model_config = ModelConfig(num_heads=num_heads,
                               num_kv_heads=num_kv_heads,
                               hidden_size=hidden_size,
                               vocab_size=vocab_size,
                               num_layers=num_layers,
                               gpt_attention_plugin=use_gpt_attention_plugin,
                               paged_kv_cache=paged_kv_cache,
                               tokens_per_block=tokens_per_block,
                               remove_input_padding=remove_input_padding,
                               dtype=dtype,
                               quant_mode=quant_mode,
                               use_custom_all_reduce=use_custom_all_reduce)

    return model_config, tp_size, pp_size, dtype, batch_size, input_len, output_len


def parse_input(inputs: List[str], tokenizer, end_id: int,
                remove_input_padding: bool):
    input_tokens = [tokenizer.encode(text, add_special_tokens=False) for text in inputs]
    input_ids = None
    input_lengths = torch.tensor([len(x) for x in input_tokens],
                                 dtype=torch.int32,
                                 device='cuda')
    if remove_input_padding:
        input_ids = np.concatenate(input_tokens)
        input_ids = torch.tensor(input_ids, dtype=torch.int32,
                                 device='cuda').unsqueeze(0)
    else:
        input_ids = torch.nested.to_padded_tensor(
            torch.nested.nested_tensor(input_tokens, dtype=torch.int32),
            end_id).cuda()

    return input_ids, input_lengths


def enc_output(output_ids, input_lengths, max_output_len, tokenizer):
    out = []
    num_beams = output_ids.size(1)
    for b in range(input_lengths.size(0)):
        for beam in range(num_beams):
            output_begin = input_lengths[b]
            output_end = input_lengths[b] + max_output_len
            outputs = output_ids[b][beam][output_begin:output_end].tolist()
            output_text = tokenizer.decode(outputs)
            out.append(output_text)
    return out




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec_id', type=str, default=None)
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--tmpdir', type=str, required=True, default=None)
    parser.add_argument('--engine_dir', type=str, default='llama_outputs')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default=".",
                        help="Directory containing the tokenizer.model.")
    parser.add_argument(
        '--output_file',
        type=str,
        help= 'output json file path',
        default=None)
    parser.add_argument(
        '--joint_to',
        type=str,
        help= 'join my result to exist json file, the result will be put into json with key specified by --name',
        default=None)
    parser.add_argument(
        '--name',
        type=str,
        help= 'test name',
        required=True,
        default=None)
    parser.add_argument(
        '--input_tokens',
        dest='input_file',
        type=str,
        help=
        'input json file, format like https://github.com/openppl-public/ppl.llm.serving/blob/master/tools/samples_1024.json',
        default=None)
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    return parser.parse_args()

def parse_json(path):
    res = {} # id of str: Dict of [q, gpt]
    with open(path, 'r') as fp:
        js = json.load(fp)
        for item in js:
            d = {}
            c = item["conversations"]
            for v in c:
                if v["from"] == "human":
                    d["q"] = v["value"]
                elif v["from"] == "gpt":
                    d["gpt"] = v["value"]
                else:
                    assert False, f'unknown key beside human and gpt: {v["from"]}'
            res[item["id"]] = d
            print(f'{item["id"]}: {d["q"]}')
    return res
                    
def generate(
    tmpdir: str = None,
    spec_id: str = None,
    name: str = None,
    log_level: str = 'error',
    engine_dir: str = None,
    input_file: str = None,
    tokenizer_dir: str = None,
    join_to: str = None,
    output_file: str = None,
    num_beams: int = 1
):
    tensorrt_llm.logger.set_level(log_level)

    if join_to is not None:
        assert os.path.exists(join_to)
    if spec_id is not None:
        output_file = None
        join_to = None

    engine_dir = Path(engine_dir)
    config_path = engine_dir / 'config.json'
    model_config, tp_size, pp_size, dtype, batch, inlen, outlen = read_config(config_path)
    world_size = tp_size * pp_size

    batch = min(16, batch)

    src = parse_json(input_file)
    if spec_id is not None:
        assert spec_id in src
        src = {spec_id: src[spec_id]}
    keys = list(src.keys())

    runtime_rank = tensorrt_llm.mpi_rank()
    runtime_mapping = tensorrt_llm.Mapping(world_size,
                                           runtime_rank,
                                           tp_size=tp_size,
                                           pp_size=pp_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir, legacy=False)

    sampling_config = SamplingConfig(end_id=EOS_TOKEN,
                                     pad_id=PAD_TOKEN,
                                     num_beams=num_beams)

    engine_name = get_engine_name('llama', dtype, tp_size, pp_size,
                                  runtime_rank)
    serialize_path = engine_dir / engine_name
    with open(serialize_path, 'rb') as f:
        engine_buffer = f.read()
    decoder = tensorrt_llm.runtime.GenerationSession(model_config,
                                                     engine_buffer,
                                                     runtime_mapping,
                                                     debug_mode=False,
                                                     debug_tensors_to_save=None)
    if runtime_rank == 0:
        print(f"Running the {dtype} engine ...")

    tmpfile = None
    if tmpdir is not None:
        tmpfile = os.path.join(tmpdir, 'chat_tmp.json')

    for i in range(0, len(keys), batch):
        print(f"Running {i}")
        focus = keys[i:i+batch]
        inputs = [src[k]["q"] for k in focus]
        input_ids, input_lengths = parse_input(inputs, tokenizer,
                                            EOS_TOKEN,
                                            model_config.remove_input_padding)

        max_input_length = torch.max(input_lengths).item()
        decoder.setup(input_lengths.size(0), max_input_length, outlen,
                    num_beams)

        output_ids = decoder.decode(input_ids,
                                        input_lengths,
                                        sampling_config,
                                        streaming=False)
        torch.cuda.synchronize()
        if runtime_rank == 0:
            out = enc_output(output_ids, input_lengths, outlen, tokenizer)
            for j, k in enumerate(focus):
                src[k][name] = out[j]
            if i % 10 == 0 and tmpfile is not None:
                with open(tmpfile, 'w') as tfp:
                    json.dump(src, tfp, ensure_ascii=False, indent=2)
    
    if runtime_rank == 0:
        written = False
        if output_file is not None:
            with open(output_file, 'w') as wp:
                json.dump(src, wp, ensure_ascii=False, indent=2)
                print(f"{name} result has been written to {output_file}")
                written = True
        
        output = src
        if join_to is not None:
            with open(join_to, 'r') as rp:
                js = json.load(rp)
                for identity in js:
                    if identity in js:
                        js[identity][name] = src[identity][name]
                with open(join_to, 'w') as wp:
                    json.dump(js, wp, ensure_ascii=False, indent=2)
                    print(f"{name} result has been joint to {join_to}")
                    written = True

        if not written:
            s = json.dumps(output, ensure_ascii=False, indent=2)
            print(s)



if __name__ == '__main__':
    args = parse_arguments()
    generate(**vars(args))
