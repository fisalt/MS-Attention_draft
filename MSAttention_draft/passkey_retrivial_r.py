# Written by Yukang Chen
# Core code based on https://github.com/CStanKonrad/long_llama
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
from numpy import random
from tqdm import tqdm
import transformers
from peft import PeftModel
# from llama_attn_replace import replace_llama_attn
from llama_attn_replace import replace_llama_attn, LlamaAttention_mss,LlamaAttention_mss_mix,LlamaAttention_mss_c,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_nope,LlamaAttention_mss_ccc_fpe,LlamaAttention_mss_ccc_a
from llama_attn_replace_o1 import forward_flashattne2,forward_flashattne,LlamaAttention_mss_ccc_spse,LlamaAttention_mss_ccc_Rma,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_R2,LlamaAttention_mss_ccc_Rt,LlamaAttention_mss_ccc_t,LlamaAttention_mss_ccc_s,LlamaAttention_mss_ccc_sps,LlamaAttention_mss_ccc_sps_mix

from llama_attn_replacemss import LlamaAttention_mss
import deepspeed

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='whether to use flash attention 2')
    parser.add_argument('--max_tokens', type=int, default=32000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    parser.add_argument('--replace_l', type=int, default=7, help='number of repeat testing for each length')
    parser.add_argument('--replace_lm', type=int, default=0, help='number of repeat testing for each length')
    parser.add_argument('--chunksize', type=int, default=16384, help='number of repeat testing for each length')
    parser.add_argument('--gpus', type=int, default=2, help='number of repeat testing for each length')
    parser.add_argument('--local_rank', type=int, default=0, help='number of repeat testing for each length')

    args = parser.parse_args()
    return args


def generate_prompt_landmark1(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 5000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def passkey_retrieval_test1(model, tokenizer, device, use_cache=False, n_garbage=60000, seed=666):
    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # input_ids = input_ids.to(device)
    len_token = input_ids.shape[-1]
    print(input_ids.shape)
    # print(input_ids)
    # print((tokenizer(tokenizer.eos_token, return_tensors="pt")))
    # print((tokenizer(tokenizer.eos_token, return_tensors="pt").input_ids[-1][-1]))
    # print(torch.tensor([2]*(16384-len_token)))
    # input_ids = torch.cat((input_ids,torch.tensor([tokenizer(tokenizer.eos_token, return_tensors="pt").input_ids[-1][-1]]*(16384-len_token))),dim=-1)
    # input_ids = torch.cat((input_ids,torch.tensor([2]*(16384-len_token))[None,:]),dim=-1)
    input_ids = input_ids.to(device)


    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
    generation_output = model.generate(
        input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
    )
    # generation_output[0]=generation_output[0,:len_token]

    model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

    is_correct = (model_answer == answer_ids[0]).all().item()
    #print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
    #print(f"The model answer is {tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}")
    return is_correct, len_token


def passkey_retrieval_test(model, tokenizer, device, use_cache=False, n_garbage=60000, seed=555):

  #n_garbage=60000 results in ~16k tokens

  prompt, answer = generate_prompt_landmark(n_garbage, seed)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
  input_ids = input_ids.cuda()
  len_token=input_ids.shape[-1]
  print(f"Prompt has {input_ids.shape[-1]} tokens")

  answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
  generation_output = model.generate(
      input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
  )

  model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

  is_correct = (model_answer == answer_ids[0]).all().item()
  print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
  print(f"The model answer is ({tokenizer.decode(model_answer.cpu())}), is_correct : {is_correct}")
#   print(f"The model generation_output is {tokenizer.decode(generation_output[0].cpu())}")
  return is_correct, len_token

import deepspeed

def main(args):
    # device = "cuda:0"
    # torch.cuda.set_device(device)

    print("base model", args.base_model)

    if args.flash_attn:
        replace_llama_attn(use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    # config._flash_attn_2_enabled=True

    context_size = args.context_size
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     args.base_model,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )
    if args.replace_lm==1:
        from llama_LM_replace2 import LlamaForCausalLM
        from modeling_llama2 import LlamaForCausalLM
    else:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    # transformers.models.llama.modeling_llama.LlamaAttention = LlamaAttention_mss
    # model = transformers.LlamaForCausalLM.from_pretrained(
    # model = LlamaForCausalLM.from_pretrained(
    #     args.base_model,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    #     chunksize=args.chunksize,
    #     max_len=args.max_tokens
    # )
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )
    # config.deepspeed= "ds_configs/stage2.json"
    # models = LlamaForCausalLM.from_pretrained(
    #     args.base_model,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    # )
    # # config.deepspeed= "ds_configs/stage3_own.json"

    # # models=model.clone()
    # # model = transformers.LlamaForCausalLM.from_pretrained(
    # #     "meta-llama/Llama-2-7b-hf",
    # #     config=config,
    # # )
    # # print("models",models.model.layers[2].self_attn.q_proj.weight)
    # # print("models",models.model.layers[0].self_attn.o_proj.weight)
    # if args.replace_l==1:
    #     replace_llama_attn(False, False,ss=5)
    # elif args.replace_l==2:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc_sps_mix(config)
    # elif args.replace_l==3:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_c(config)
    # elif args.replace_l==4:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc_sps(config)
    # elif args.replace_l==5:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc(config)
    # elif args.replace_l==6:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc_t(config)
    #         # LlamaAttention_mss_ccc_nope
    # elif args.replace_l==7:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc_Rt(config)
    #         # LlamaAttention_mss_ccc_nope
    # elif args.replace_l==8:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc_sps(config)
    # elif args.replace_l==9:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss_ccc_spse(config)
    # else:
    #     transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattne
    #         # model.model.layers[i].self_attn.q_proj.weight=models.model.layers[i].self_attn.q_proj.weight
    #         # model.model.layers[i].self_attn.k_proj.weight=models.model.layers[i].self_attn.k_proj.weight
    #         # model.model.layers[i].self_attn.v_proj.weight=models.model.layers[i].self_attn.v_proj.weight
    #         # model.model.layers[i].self_attn.o_proj.weight=models.model.layers[i].self_attn.o_proj.weight

    # # model.load_state_dict(model_o.state_dict(),strict=False)
    # # replace_llama_attn(False, False,ss=0)
    # model.load_state_dict(models.state_dict(),strict=False)
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattne
    model.resize_token_embeddings(32001)
    deepspeed_config = "ds_configs/stage3_own2.json"

    ds_model = deepspeed.init_inference(
        model=model,      # Transformers模型
        mp_size=args.gpus,        # GPU数量
        # dtype=torch.bfloat16, # 权重类型(fp16)
        dtype=torch.float16, # 权重类型(fp16)
        replace_method="auto", # 让DS自动替换层
        # replace_with_kernel_inject=True, # 使用kernel injector替换
        max_tokens=args.max_tokens
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        # n_garbage = int(15* 4 * (i + 1) * args.interval // 1024 * 1024)
        # n_garbage = 1024
        passed_tests = 0
        total_tokens = 0
        for i in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval_test(ds_model, tokenizer, None, use_cache=False, n_garbage=n_garbage, seed=i)
            passed_tests += is_correct
            total_tokens += len_tokens
        avg_tokens = total_tokens//args.num_tests
        accuracy = float(passed_tests)/args.num_tests
        print("accuracy on the token length %d is %f"%(avg_tokens, accuracy))
        all_accuries[str(avg_tokens)] = accuracy
    print("accuries over tokens", all_accuries)


if __name__ == "__main__":
    args = parse_config()
    main(args)
