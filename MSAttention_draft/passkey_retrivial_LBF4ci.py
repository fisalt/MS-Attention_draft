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


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='whether to use flash attention 2')
    parser.add_argument('--max_tokens', type=int, default=32000, help='maximum token length for evaluation')
    parser.add_argument('--interval', type=int, default=1000, help='interval for evaluation')
    parser.add_argument('--num_tests', type=int, default=5, help='number of repeat testing for each length')
    parser.add_argument('--replace_l', type=int, default=7, help='number of repeat testing for each length')
    parser.add_argument('--replace_lm', type=int, default=0, help='number of repeat testing for each length')
    parser.add_argument('--local_rank', type=int, default=0, help='number of repeat testing for each length')
    parser.add_argument('--chunksize', type=int, default=16384, help='number of repeat testing for each length')
    parser.add_argument('--scaling_factor', type=float, default=8, help='number of repeat testing for each length')

    args = parser.parse_args()
    return args

args = parse_config()

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
                args.scaling_factor
                # (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
                # 256
                # (4096 * 4) - (4096 - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin
    

class LlamaDynamicNTKScalingRotaryEmbeddingbyPart(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        # seq_len = 16384
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                # base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
                base ** ((torch.arange(0, self.dim, 2, dtype=torch.int64)).float().to(x.device) / self.dim)
            )
            inv_freq=torch.cat((self.inv_freq[:self.dim//4],inv_freq[self.dim//4:]))
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin

class LlamaDynamicNTKScalingRotaryEmbeddingMod(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        # position_ids=position_ids%16384
        # position_ids=position_ids%8192
        # seq_len = torch.max(position_ids) + 1
        # seq_len = 16384
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                # (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
                args.scaling_factor
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin

class LlamaDynamicNTKScalingRotaryEmbeddingModInf(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        # position_ids=position_ids%16384
        # position_ids=position_ids%8192
        # seq_len = torch.max(position_ids) + 1
        # seq_len = 16384
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                # (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
                args.scaling_factor
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        cos=cos+0.5
        return cos, sin

def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 500000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 5000)
    pass_key = random.randint(1, 99)
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





def passkey_retrieval_test(model, tokenizer, device, use_cache=False, n_garbage=60000, seed=555):

  #n_garbage=60000 results in ~16k tokens

  prompt, answer = generate_prompt_landmark(n_garbage, seed)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#   input_ids = input_ids.to(device)
  input_ids = input_ids.cpu()
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
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)



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
    if args.replace_lm==1:
        # from llama_LM_replace2 import LlamaForCausalLM
        from modeling_llama2 import LlamaForCausalLM
    else:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = LlamaLinearScalingRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbeddingbyPart
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbeddingMod
    transformers.models.llama.modeling_llama.LlamaDynamicNTKScalingRotaryEmbedding = LlamaDynamicNTKScalingRotaryEmbeddingModInf
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
    # import deepspeed
    # from accelerate import init_empty_weights

    # with init_empty_weights():
    #     model = LlamaForCausalLM.from_pretrained(
    #         args.base_model,
    #         config=config,
    #         cache_dir=args.cache_dir,
    #         torch_dtype=torch.bfloat16,
    #     )
        
    # from accelerate import infer_auto_device_map

    # device_map = infer_auto_device_map(model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})

    # from accelerate import load_checkpoint_and_dispatch

    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=args.base_model, device_map="balanced_low_0", no_split_module_classes=['LlamaDecoderLayer','layers']
    # ).to(torch.bfloat16)
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
    from llama_LM_replace_R import forward_LMt,forward_LMc,forward_LMca
    from llama_attn_replace_o2 import forward_LBc
    transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = _prepare_decoder_attention_mask
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_LBe
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_LBc
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward_LMc
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = forward_LMca
    # transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_ring_flashattne
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
    for i in range(total_test_points):
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * (i + 1) * args.interval // 1024 * 1024)
        # n_garbage = int(15* 4 * (i + 1) * args.interval // 1024 * 1024)
        # n_garbage = 1024
        passed_tests = 0
        total_tokens = 0
        for i in range(args.num_tests):
            is_correct, len_tokens = passkey_retrieval_test(model.cpu().to(torch.bfloat16), tokenizer, device, use_cache=False, n_garbage=n_garbage, seed=i)
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
