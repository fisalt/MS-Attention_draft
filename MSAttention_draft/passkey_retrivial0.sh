#!/bin/bash
STEPS=5000
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
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyPI_dyqek_m64_w16s512_zkojobPI32K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyPI_dyqek_m64_w16s512_zkojobPI32K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale8nodyPI_dyqek_m64_w16s512_zkojobPI32K_acc4_g4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_bypart/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_zeropart/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod8K/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32K/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmax32K/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmax32Ku2/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmax50K_testqk/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmax32K/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1M/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MinfiC/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MZ4/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MinfiC1e-1/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MinfiC5e-1/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_ModDynamicS8Ka16K/checkpoint-$STEPS"
MODEL_PATH="path_to_saving_checkpoints16k_scale16384nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32K_Zgit/checkpoint-$STEPS"
cd $MODEL_PATH/global_step$STEPS
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
CUDA_VISIBLE_DEVICES=0 python3 merge_lora_weights_and_save_hf_model.py \
        --base_model /home/wangning/transformer-xl-master/output/llama \
        --peft_model $MODEL_PATH \
        --context_size 65536 \
        --save_path /home/wangning/LongLoRA-main/$MODEL_PATH-7b-longlora-kvcc-merged

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LB.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 68008 \
        --interval 8000 \
        --replace_l 8   \
        --scaling_factor 1024   \
        --replace_lm 0   \
        --dynamic 1

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LB.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 68008 \
        --interval 8000 \
        --replace_l 8   \
        --scaling_factor 4096   \
        --replace_lm 0   \
        --dynamic 1

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LB.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 68008 \
        --interval 8000 \
        --replace_l 8   \
        --scaling_factor 16384   \
        --replace_lm 0   \
        --dynamic 1