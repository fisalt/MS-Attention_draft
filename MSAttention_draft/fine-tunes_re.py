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
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from llama_attn_replace import replace_llama_attn, LlamaAttention_mss,LlamaAttention_mss_mix,LlamaAttention_mss_c,LlamaAttention_mss_cc,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_nope,LlamaAttention_mss_ccc_fpe,LlamaAttention_mss_ccc_a
from gptneox_attn_replace import replace_gpt_neox_attn
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
import llama_attn_replace_0

from llama_attn_replace_o1 import LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_R2,LlamaAttention_mss_ccc_Rmax,LlamaAttention_mss_ccc_Rmamm,LlamaAttention_mss_ccc_Rmam,LlamaAttention_mss_ccc_Rm,LlamaAttention_mss_ccc_Rmd,LlamaAttention_mss_ccc_Rma,LlamaAttention_mss_ccc_Rmb,LlamaAttention_mss_ccc_Rmc
from llama_LM_replace_R import CustomTrainer
from datasets import load_dataset

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

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

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

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # NOTE: May expand supported model types in the future
    if model_args.model_type == "gpt-neox":
        replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    else:
        assert model_args.model_type == "llama", "Only support llama and gpt-neox for now"
        if training_args.replace_l==0:
            llama_attn_replace_0.replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)
        else:
            # replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)
            pass

    # transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention_mss
    # sys.modules['transformers.models.llama.modeling_llama'].LlamaAttention = LlamaAttention_mss
    # Set RoPE scaling factor
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
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    if training_args.replace_lm==1:
        from llama_LM_replace_R import LlamaForCausalLM
    else:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    # transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention_mss
    # model = transformers.LlamaForCausalLM.from_pretrained(
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        chunksize=training_args.chunksize,
        max_len=training_args.model_max_length
    )
    # config.deepspeed= "ds_configs/stage2.json"
    models = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        chunksize=training_args.chunksize,
        max_len=training_args.model_max_length
    )
    # config.deepspeed= "ds_configs/stage3_own.json"

    # models=model.clone()
    # model = transformers.LlamaForCausalLM.from_pretrained(
    #     "meta-llama/Llama-2-7b-hf",
    #     config=config,
    # )
    # print("models",models.model.layers[2].self_attn.q_proj.weight)
    # print("models",models.model.layers[0].self_attn.o_proj.weight)
    if training_args.replace_l==2:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_mix(config)
    elif training_args.replace_l==3:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_c(config)
    elif training_args.replace_l==4:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_cc(config)
    elif training_args.replace_l==5:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rmax(config,fit_func_list=model.trans)
    elif training_args.replace_l==6:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rmam(config,fit_func_list=model.trans)
            # model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rm(config,fit_func=model.fit_func)
            # LlamaAttention_mss_ccc_nope
    elif training_args.replace_l==7:
        for i in range(len(model.model.layers)):
            # model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rmc(config,fit_func=model.fit_func)
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rmamm(config,fit_func_list=model.trans)
            # LlamaAttention_mss_ccc_nope
    elif training_args.replace_l==8:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rma(config,fit_func_list=model.trans)
    else:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss(config)
            # model.model.layers[i].self_attn.q_proj.weight=models.model.layers[i].self_attn.q_proj.weight
            # model.model.layers[i].self_attn.k_proj.weight=models.model.layers[i].self_attn.k_proj.weight
            # model.model.layers[i].self_attn.v_proj.weight=models.model.layers[i].self_attn.v_proj.weight
            # model.model.layers[i].self_attn.o_proj.weight=models.model.layers[i].self_attn.o_proj.weight

    # model.load_state_dict(model_o.state_dict(),strict=False)
    model.load_state_dict(models.state_dict(),strict=False)
   
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    # )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=training_args.cache_dir)
    dataset = dataset.map(partial(tokenize_fn,tokenizer),batched=True, num_proc=64, remove_columns=["text", "meta"])

    if rank == 0:
        barrier()

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if training_args.low_rank_training:
        if model_args.model_type == "gpt-neox":
            # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
            targets = ["query_key_value", "dense"]
        else:
            targets=["q_proj", "k_proj", "v_proj", "o_proj"]
            # targets=None
            # targets=["q_proj", "v_proj"]
            # targets=["o_proj"]

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=targets,
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # enable trainable params
        # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")]) or "embed" in n or "norm" in n or "fit_func" in n or "trans" in n]
        # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")]) or "embed" in n or "norm" in n or "fit_func" in n or "trans" in n or "k_proj" in n or "o_proj" in n]
        # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")]) or "embed" in n or "norm" in n]

    # for n, p in model.named_parameters():
    #     if "embed" in n or "norm" in n or "fit_func" in n or "trans" in n:
    #     # if "embed" in n or "norm" in n:
    #         p.requires_grad=True
    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = CustomTrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
