# Written by Yukang Chen
# Some code based on https://github.com/epfml/landmark-attention
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import torch
import argparse
import random
import numpy as np
from tqdm import tqdm
import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from peft import PeftModel
# from llama_attn_replace import replace_llama_attn
from llama_attn_replace import replace_llama_attn, LlamaAttention_mss,LlamaAttention_mss_mix,LlamaAttention_mss_c,LlamaAttention_mss_ccc_nope
# from llama_attn_replace_o1 import LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_R2,LlamaAttention_mss_cc,
from llama_attn_replace_o1 import forward_LBe,forward_flashattne,LlamaAttention_mss_cc1,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_R2,LlamaAttention_mss_ccc_Rt,LlamaAttention_mss_ccc_t,LlamaAttention_mss_ccc_s,LlamaAttention_mss_ccc_sps,LlamaAttention_mss_ccc_sps_mix
from llama_attn_replacemss import LlamaAttention_mss
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size during inference')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=2048, help='context length during evaluation')
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--peft_model', type=str, default=None, help='')
    parser.add_argument('--flash_attn', type=bool, default=True, help='')
    parser.add_argument('--data_path', type=str, default="./test.bin", help='')
    parser.add_argument('--sliding_window', type=int, default=16384, help='')
    parser.add_argument('--attn_select', type=int, default=0, help='')
    parser.add_argument('--lm_select', type=int, default=0, help='')
    args = parser.parse_args()
    return args

args = parse_config()
def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}
from torch.utils.data import Dataset, DataLoader
# 自定义数据集类
class StreamDataset(Dataset):
    def __init__(self, dataset, tokenizer, block_size=32768):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.current_buffer = []
        self.buffer_size = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 动态加载并拼接数据
        while self.buffer_size < self.block_size:
            sample = self.dataset[idx]
            tokenized_sample = self.tokenizer(sample['text'], return_special_tokens_mask=True)
            self.current_buffer.extend(tokenized_sample['input_ids'])
            self.buffer_size += len(tokenized_sample['input_ids'])
            idx += 1
            if idx >= len(self.dataset):
                break

        # 截断到block_size
        input_ids = self.current_buffer[:self.block_size]
        labels = input_ids.copy()

        # 更新缓冲区
        self.current_buffer = self.current_buffer[self.block_size:]
        self.buffer_size -= self.block_size

        return {"input_ids": input_ids, "labels": labels}





class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin
    
class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin


import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
import json
from transformers.models.llama.modeling_llama import LlamaForCausalLM
transformers.models.llama.modeling_llama.LlamaRotaryEmbedding=LlamaRotaryEmbedding
transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding=LlamaLinearScalingRotaryEmbedding
from llama_attn_replace_o1 import Mistral_mss_ccc_spse
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
transformers.models.mistral.modeling_mistral.MistralAttention=Mistral_mss_ccc_spse
if True:
    config = transformers.AutoConfig.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    torch_dtype=torch.float16
    model = MistralForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir="./mistralai",
        config=config,
        torch_dtype=torch.bfloat16,
        # chunksize=training_args.chunksize,
        # max_len=training_args.model_max_length
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        cache_dir="./mistralai",
        model_max_length=args.seq_len,
        padding_side="right",
        use_fast=True,
    )
    
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     args.base_model,
    #     cache_dir=args.cache_dir,
    #     model_max_length=args.seq_len,
    #     padding_side="right",
    #     use_fast=True,
    # )
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(32001)

    if args.peft_model:
        trainable_params = os.path.join(args.peft_model, "trainable_params.bin")
        if os.path.isfile(trainable_params):
            model.load_state_dict(torch.load(trainable_params, map_location=model.device), strict=False)
        else:
            raise ValueError("Trainable input embedding and normalization are required.")
        model = PeftModel.from_pretrained(
            model,
            args.peft_model,
            device_map="auto",
            torch_dtype=torch_dtype,
        )

# 加载模型和tokenizer
# torch_dtype=torch.float16
# model_path = "/home/wangning/transformer-xl-master/output/llama"
# model = LlamaForCausalLM.from_pretrained(model_path,torch_dtype=torch_dtype)
# tokenizer = LlamaTokenizer.from_pretrained(model_path)
transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattne
# transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_LBe
# 确保模型在GPU上（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载Proof-Pile数据集
# dataset = load_dataset("json", data_files={"test": "./proofpile_test.jsonl"})["test"]
dataset = load_dataset('pg19',cache_dir="/home/wangning/pg19/datasets")["test"]

def compute_ppl(model, tokenizer, dataset, max_length):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    i=0

    with torch.no_grad():
        for data in dataset:
            inputs = tokenizer(data["text"], return_tensors="pt", truncation=True, max_length=max_length, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            print(loss)

            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)
            if i>1999:
                break
            i+=1

    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

# 计算不同长度的PPL
lengths = [32768, 16384, 8192, 4096]
lengths = [4096, 8192, 16384, 32768]
results = {}

for length in lengths:
    ppl = compute_ppl(model, tokenizer, dataset, max_length=length)
    results[length] = ppl
    print(f"Length {length}: PPL = {ppl}")

    # 输出结果
    with open("ppl_results_PG.json", "w") as f:
        json.dump(results, f, indent=4)

    print("PPL results:", results)

