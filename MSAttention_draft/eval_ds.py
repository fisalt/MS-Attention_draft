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
from llama_attn_replace import replace_llama_attn
from llama_attn_replace import replace_llama_attn, LlamaAttention_mss,LlamaAttention_mss_mix,LlamaAttention_mss_c,LlamaAttention_mss_cc,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_nope
from llama_attn_replace_o1 import forward_flashattne,LlamaAttention_mss_ccc,LlamaAttention_mss_ccc_R2

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
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--gpus', type=int, default=0, help='')
    args = parser.parse_args()
    return args

def get_as_batch(data, seq_length, batch_size, device='cpu', sliding_window=256):
    all_ix = list(range(0, len(data) - seq_length, sliding_window))
    all_ix.pop()

    for idx in range(0, len(all_ix), batch_size):
        ix = all_ix[idx:idx+batch_size]
        assert all([idx + seq_length + 1 <= len(data) for idx in ix])
        x = torch.stack([torch.from_numpy((data[i:i+seq_length]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_length]).astype(np.int64)) for i in ix])
        if device != 'cpu':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        yield x, y

def iceildiv(x, y):
    return (x + y - 1) // y

def evaluate(model, data, batch_size, device, seq_length, sliding_window=256, use_cache=False):
    stats = {}

    model.eval()

    loss_list_val, acc_list = [], []
    loss_step_list_val = []

    with torch.no_grad():
        print(f"Using seq length {seq_length}")
        torch.set_printoptions(sci_mode=False)
        for idx, (x, y) in tqdm(
            enumerate(
                get_as_batch(
                    data['val'], 
                    seq_length, 
                    batch_size, 
                    device=device,
                    sliding_window=sliding_window
                )
            ),
            total=iceildiv(
                iceildiv(len(data['val']), sliding_window),
                batch_size
            )
        ):
            val_loss = 0.
            acc = 0.
            cnt = 0

            for part_idx, i in enumerate(range(0, x.shape[1], seq_length)):
                part_len = x[:, i:i + seq_length].shape[1]

                outputs = model(
                    input_ids=x[:, i:i + seq_length].cuda(),
                    labels=x[:, i:i+seq_length].contiguous().cuda(),
                    use_cache=use_cache)

                val_loss = outputs.loss * part_len + val_loss
                acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length].cuda()).float().sum()) + acc
                # acc = ((outputs.logits.argmax(-1) == y[:, i:i+seq_length]).float().sum()) + acc
                cnt += part_len
                while len(loss_step_list_val) <= part_idx:
                    loss_step_list_val.append([])
                loss_step_list_val[part_idx].append(outputs.loss.item())
            val_loss /= cnt
            acc /= cnt
            
            loss_list_val.append(val_loss.item())
            acc_list.append(acc.item())

            # if idx>50:
            #     break

    stats['val_acc'] = torch.as_tensor(acc_list).mean().item()
    stats['val_loss'] = torch.as_tensor(loss_list_val).mean().item()
    stats['val_perplexity'] = 2.71828 ** stats['val_loss']
    stats['val_perplexity_per_chunk'] = torch.exp(torch.as_tensor(loss_step_list_val).mean(dim=1))

    return stats

def main(args):

    device = "cuda:0"
    # device = "cuda"
    seed = 2
    torch.cuda.set_device(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = {'val': np.memmap(args.data_path, dtype=np.uint16, mode='r')}

    print(f"Num validation tokens: {len(data['val'])}")
    print("data path", args.data_path)
    print("base model", args.base_model)
    print("peft model", args.peft_model)

    # if args.flash_attn:
    #     replace_llama_attn(use_flash_attn=True, use_full=True)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(config, "max_position_embeddings", None) # this value should be 4096 for LLaMA2 models
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    if args.lm_select==1:
        from llama_LM_replace_R import LlamaForCausalLM
    else:
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    torch_dtype=torch.float16
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype)
    models = LlamaForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch_dtype)
    # model.resize_token_embeddings(32001)
    if args.attn_select==2:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_mix(config)
    elif args.attn_select==3:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_c(config)
    elif args.attn_select==4:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_cc(config)
    elif args.attn_select==5:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc(config)
    elif args.attn_select==6:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc_nope(config)
    elif args.attn_select==7:
        for i in range(len(model.model.layers)):
            model.model.layers[i].self_attn=LlamaAttention_mss_ccc_R2(config)
    # else:
    #     for i in range(len(model.model.layers)):
    #         model.model.layers[i].self_attn=LlamaAttention_mss(config)
    else:
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_flashattne

    model.load_state_dict(models.state_dict(),strict=False)
    # for i in range(len(model.model.layers)):
    #     # print(i)
    #     model.model.layers[i].self_attn=model.model.layers[i].self_attn.to(torch.bfloat16)
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
    # from deepspeed import DeepSpeedEngine
    # ds_engine = DeepSpeedEngine()
    # model = ds_engine.initialize_model(model=model, num_gpus=2)
    model = model.merge_and_unload()
    import deepspeed
    ds_model = deepspeed.init_inference(
        model=model,      # Transformers模型
        mp_size=args.gpus,        # GPU数量
        # dtype=torch.bfloat16, # 权重类型(fp16)
        dtype=torch.float16, # 权重类型(fp16)
        replace_method="auto", # 让DS自动替换层
        # replace_with_kernel_inject=True, # 使用kernel injector替换
    )


    # stats = evaluate(model.cuda().to(torch.bfloat16), data, args.batch_size, device, args.seq_len, sliding_window=16384)
    # stats = evaluate(model.cuda().to(torch.bfloat16), data, args.batch_size, device, args.seq_len, sliding_window=args.sliding_window)
    stats = evaluate(ds_model, data, args.batch_size, device, args.seq_len, sliding_window=args.sliding_window)

    print(stats)


if __name__ == "__main__":
    args = parse_config()
    main(args)
