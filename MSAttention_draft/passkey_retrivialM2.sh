#!/bin/bash
STEPS=3000
LEN=16384
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojob/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale16_m64_w16s512_kojobntk_acc4_g4/checkpoint-3000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale1_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
# MODEL_PATH="path_to_saving_checkpoints16k_scale64_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk_acc4_g4/checkpoint-3500"
MODEL_PATH="path_to_saving_checkpoints16k_scale128_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
# MODEL_PATH="path_to_saving_checkpoints16k_scale512_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale2048_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16384_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale65536_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16384_m64_w16s512_zkojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobPI_acc4_g4/checkpoint-$STEPS"
# MODEL_PATH="path_to_saving_checkpoints16k_scale16dyft_m64_w16s512_kojobntk_acc4_g4/checkpoint-$STEPS"
# MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyntk_dyqek_m64_w16s512_zkojobPI128K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale64nodyntk_dyqek_m64_w16s512_zkojobPI256K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyntk_dyqek_m64_w16s512_zkojobPI128K__acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale1632dypi_dyqek_m64_w16s512_zkojobPI128K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyntk_dyqek_m64_w16s512_zkojobPI128K__acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyntk_dyqek_m64_w16s512_zkojobPI128K__acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyntk_dypos_m64_w16s512_zkojobntk_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyNTK_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="Mpath_to_saving_checkpoints16k_glolen_m64_w16s512_kojobNTK/checkpoint-$STEPS"
# MODEL_PATH="Mpath_to_saving_checkpoints16k_scale16_m64_w16s512_kojobNTK/checkpoint-1500"
# MODEL_PATH="Mpath_to_saving_checkpoints16k_scale64_m64_w16s512_kojobNTK/checkpoint-5000"
MODEL_PATH="Mpath_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobNTK/checkpoint-4000"
# cd $MODEL_PATH/global_step$STEPS
# # 创建Python脚本
# cat <<EOL > extract_parameters.py
# import torch

# # 加载模型状态
# model_state = torch.load("mp_rank_00_model_states.pt")

# # 提取模型参数
# model_dict = model_state['module']

# # 保存模型参数到新的文件
# torch.save(model_dict, "pytorch_model.bin")
# EOL

# # 运行Python脚本
# python extract_parameters.py
# mv pytorch_model.bin ../pytorch_model.bin
# cd ..
# cd ..
# cd ..
# python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,o_proj,k_proj"
# CUDA_VISIBLE_DEVICES=0 python3 merge_lora_weights_and_save_hf_modelM2.py \
#         --base_model /home/wangning/transformer-xl-master/output/llama \
#         --peft_model $MODEL_PATH \
#         --context_size 16384 \
#         --save_path /home/wangning/LongLoRA-main/$MODEL_PATH-7b-longlora-kvcc-merged
CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBM.py \
        --context_size 16384 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 68008 \
        --interval 56789 \
        --num_tests 6   \
        --replace_l 8   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBM.py \
        --context_size 16384 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 68008 \
        --interval 56789 \
        --replace_l 8   \
        --replace_lm 0