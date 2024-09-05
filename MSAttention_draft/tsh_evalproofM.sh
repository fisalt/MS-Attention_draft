#!/bin/bash
# MODEL_PATH="path_to_saving_checkpoints32k_z2off-m128-512r16-ko/checkpoint-5000"
# MODEL_PATH="Mpath_to_saving_checkpoints16k_glolen_m64_w16s512_kojob3/checkpoint-5000"
# cd $MODEL_PATH
# python zero_to_fp32.py . pytorch_model.bin
# cd ..
# cd ..
# python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,k_proj,o_proj"
MODEL_PATH="Mpath_to_saving_checkpoints16k_glolen_m64_w16s512_kojobNTK/checkpoint-3000"
cd $MODEL_PATH/global_step3000
# 创建Python脚本
cat <<EOL > extract_parameters.py
import torch

# 加载模型状态
model_state = torch.load("mp_rank_00_model_states.pt")

# 提取模型参数
model_dict = model_state['module']

# 保存模型参数到新的文件
torch.save(model_dict, "pytorch_model.bin")
EOL

# 运行Python脚本
python extract_parameters.py
mv pytorch_model.bin ../pytorch_model.bin
cd ..
cd ..
cd ..
python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,o_proj,k_proj"
CUDA_VISIBLE_DEVICES=0 python3 eval_proofM.py --seq_len 16384 --context_size 16384 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval_proofM.py --seq_len 4096 --context_size 16384 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval_proofM.py --seq_len 8192 --context_size 16384 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
# CUDA_VISIBLE_DEVICES=0 python3 eval_proofM.py --seq_len 16384 --context_size 16384 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval_proofM.py --seq_len 32768 --context_size 16384 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 16384 --context_size 32768 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 8192 --context_size 32768 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 4096 --context_size 32768 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 2048 --context_size 32768 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
# CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 65536 --context_size 32768 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0