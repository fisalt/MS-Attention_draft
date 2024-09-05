#!/bin/bash
MODEL_PATH="path_to_saving_checkpoints32k_z2off-m128-512r16-ko/checkpoint-5000"
cd $MODEL_PATH
python zero_to_fp32.py . pytorch_model.bin
cd ..
cd ..
python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,k_proj,o_proj"

# python3 eval.py --seq_len 32768 --context_size 65536 --sliding_window 32768 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path ./proofpile_test.jsonl --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 16384 --context_size 32768 --sliding_window 16384 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path pg19/test.bin --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 8192 --context_size 32768 --sliding_window 8192 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path pg19/test.bin --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 4096 --context_size 32768 --sliding_window 4096 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path pg19/test.bin --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 2048 --context_size 32768 --sliding_window 2048 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path pg19/test.bin --attn_select 0 --lm_select 0
CUDA_VISIBLE_DEVICES=0 python3 eval.py --seq_len 65536 --context_size 32768 --sliding_window 65536 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path pg19/test.bin --attn_select 0 --lm_select 0