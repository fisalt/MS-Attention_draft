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

from llama_attn_replace_o2 import forward_ring_flashattne,forward_LBe,forward_flashattne2,forward_flashattne,LlamaAttention_mss_ccc_spse,LlamaAttention_mss_ccc_Rma,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_R2,LlamaAttention_mss_ccc_Rt,LlamaAttention_mss_ccc_t,LlamaAttention_mss_ccc_s,LlamaAttention_mss_ccc_sps,LlamaAttention_mss_ccc_sps_mix
from llama_attn_replace_o1 import Mistral_mss_ccc_spse,forward_flashattnM


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
    parser.add_argument('--local_rank', type=int, default=0, help='number of repeat testing for each length')
    parser.add_argument('--dynamic', type=int, default=1, help='number of repeat testing for each length')
    parser.add_argument('--chunksize', type=int, default=16384, help='number of repeat testing for each length')

    args = parser.parse_args()
    return args


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
                # (512 * 4) - (512 - 1)
                # 256
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin
    
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
    garbage_inf = " ".join([garbage] * 50000)
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
  input_ids = input_ids.to(0)
  len_token=input_ids.shape[-1]
  print(f"Prompt has {input_ids.shape[-1]} tokens")

  answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
  generation_output = model.generate(
      input_ids=input_ids, max_new_tokens=answer_ids.shape[-1]+1, num_beams=1, use_cache=use_cache
  )

  model_answer = generation_output[0, -answer_ids.shape[-1]:].cpu()

  is_correct = (model_answer == answer_ids[0]).all().item()
  print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}")
  print(f"The model answer is ({tokenizer.decode(model_answer.cpu())}), is_correct : {is_correct}")
#   print(f"The model generation_output is {tokenizer.decode(generation_output[0].cpu())}")
  return is_correct, len_token

def _prepare_decoder_attention_mask(self, attention_mask, input_shape,
                                    inputs_embeds, past_key_values_length):
    # [bsz, seq_len]
    if input_shape[-1] > 1 and past_key_values_length == 0:  # encode
        return attention_mask
    return transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask(self, attention_mask,
                                                                                              input_shape,
                                                                                              inputs_embeds,
                                                                                              past_key_values_length)
def main(args):
    # device = "cuda:0"
    # torch.cuda.set_device(device)

    print("base model", args.base_model)
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM



    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )
    # config._flash_attn_2_enabled=True

    context_size = args.context_size
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        if args.dynamic==0:
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        else:
            config.rope_scaling = {"type": "dynamic", "factor": scaling_factor}
        # config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     args.base_model,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     torch_dtype=torch.float16,
    #     device_map="auto",
    # )

    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    transformers.models.mistral.modeling_mistral.MistralModel._update_causal_mask = _prepare_decoder_attention_mask
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattnM
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward_flashattnM
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_LBe
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_ring_flashattne
    # model = transformers.LlamaForCausalLM.from_pretrained(
    # model = LlamaForCausalLM.from_pretrained(
    #     args.base_model,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    #     chunksize=args.chunksize,
    #     max_len=args.max_tokens
    # )
    # model = LlamaForCausalLM.from_pretrained(
    #     args.base_model,
    #     config=config,
    #     cache_dir=args.cache_dir,
    #     torch_dtype=torch.bfloat16,
    # )
    import deepspeed
    from accelerate import init_empty_weights

    with init_empty_weights():
        model = MistralForCausalLM.from_pretrained(
            args.base_model,
            cache_dir=args.cache_dir,
            config=config,
            torch_dtype=torch.bfloat16,
            # chunksize=training_args.chunksize,
            # max_len=training_args.model_max_length
        )
        
    # from accelerate import infer_auto_device_map

    # device_map = infer_auto_device_map(model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
    for i in range(len(model.model.layers)):
        # model.model.layers[i].self_attn = Mistral_mss_ccc_spse(config)
        model.model.layers[i].self_attn.rotary_emb = LlamaLinearScalingRotaryEmbedding(
        # model.model.layers[i].self_attn.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
            config.hidden_size/config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=16,
            base=config.rope_theta,
        ).to(torch.bfloat16)

    from accelerate import load_checkpoint_and_dispatch
        # self.layers = nn.ModuleList([MistralDecoderLayer(config) for _ in range(config.num_hidden_layers)])

    model = load_checkpoint_and_dispatch(
        model, checkpoint=args.base_model, device_map="balanced_low_0", no_split_module_classes=['MistralDecoderLayer','layers']
    ).to(torch.bfloat16)

    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=args.base_model, device_map="balanced_low_0",max_memory = {0: "8GIB", 1: "8GIB", 2: "8GIB", 3: "8GIB"}
    # ).to(torch.bfloat16)

    # # model.load_state_dict(model_o.state_dict(),strict=False)
    # # replace_llama_attn(False, False,ss=0)
    # model.load_state_dict(models.state_dict(),strict=False)
    for i in range(len(model.model.layers)):
        # model.model.layers[i].self_attn = Mistral_mss_ccc_spse(config)
        # model.model.layers[i].self_attn.rotary_emb = LlamaLinearScalingRotaryEmbedding(
        model.model.layers[i].self_attn.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
            config.hidden_size/config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            scaling_factor=2,
            base=config.rope_theta,
        )
    # transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_LBe
    # # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_ring_flashattne
    # transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    # transformers.models.mistral.modeling_mistral.MistralModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattnM
    transformers.models.mistral.modeling_mistral.MistralAttention.forward = forward_flashattnM
    model.resize_token_embeddings(32001)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="right",
        use_fast=False,
    )

    total_test_points = args.max_tokens // args.interval
    all_accuries = {}
    with torch.no_grad():
        for i in range(total_test_points):
            # This is a rough ratio to control the number of texts and tokens
            n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
            # n_garbage = int(15* 4 * (i + 1) * args.interval // 1024 * 1024)
            # n_garbage = 1024
            passed_tests = 0
            total_tokens = 0
            for i in range(args.num_tests):
                is_correct, len_tokens = passkey_retrieval_test(model.to(torch.bfloat16), tokenizer, None, use_cache=False, n_garbage=n_garbage, seed=i)
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
