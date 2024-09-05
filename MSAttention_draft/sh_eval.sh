#!/bin/bash
MODEL_PATH="path_to_saving_checkpoints16kppl_m64-512r16-ko/checkpoint-500"
cd $MODEL_PATH
python zero_to_fp32.py . pytorch_model.bin
cd ..
cd ..
python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,k_proj,o_proj"

CUDA_VISIBLE_DEVICES=1 python3 eval.py --seq_len 16384 --context_size 16384 --sliding_window 16384 --batch_size 1 --base_model /home/wangning/transformer-xl-master/output/llama --peft_model $MODEL_PATH --data_path pg19/test.bin --attn_select 0 --lm_select 0