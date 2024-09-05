import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import transformers
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        # default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    replace_l: int = field(
        default=1,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    replace_lm: int = field(
        default=0,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    chunksize: int = field(
        default=16384,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    seg: int = field(
        default=16,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    merge: int = field(
        default=128,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    top: int = field(
        default=256,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
model_args, training_args = parser.parse_args_into_dataclasses()

config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

orig_rope_scaling = getattr(config, "rope_scaling", None)
if orig_rope_scaling is None:
    orig_rope_scaling = {"factor": 1}

orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
orig_ctx_len = getattr(config, "max_position_embeddings", None)
if orig_ctx_len:
    orig_ctx_len *= orig_rope_scaling_factor
    if training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        # config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
# 加载模型和 tokenizer
model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        # chunksize=training_args.chunksize,
        # max_len=training_args.model_max_length
    )
from llama_attn_replace_o1 import forward_flashattn,forward_LBe
transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_LBe
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=training_args.cache_dir,
    model_max_length=training_args.model_max_length,
    padding_side="right",
    use_fast=True,
)
model.eval()
model.to('cuda')


# 加载 Proof Pile 数据集
dataset = load_dataset("proof-pile")

# 准备输入长度
input_lengths = [4096, 8192, 16384, 32768]

# 计算 Perplexity 的函数
def calculate_perplexity(model, tokenizer, text, max_length):
    stride = 512  # 设置滑动窗口的步长
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to('cuda')

    max_length = min(max_length, input_ids.size(1))

    nlls = []
    for i in range(0, input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, input_ids.size(1))
        trg_len = end_loc - i
        input_ids = input_ids[:, begin_loc:end_loc].to('cuda')
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        
        log_likelihood = outputs.loss * trg_len
        nlls.append(log_likelihood)
    
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

# 遍历不同长度并计算 Perplexity
for length in input_lengths:
    sample_text = dataset['test'][0]['text'][:length]  # 取样本文本
    ppl = calculate_perplexity(model, tokenizer, sample_text, length)
    print(f"Perplexity for length {length}: {ppl}")
