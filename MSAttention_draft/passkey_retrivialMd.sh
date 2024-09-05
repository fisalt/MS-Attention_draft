#!/bin/bash
MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale16_m64_w16s512_kojobntk_acc4_g4/checkpoint-3000"
# MODEL_PATH="path_to_saving_checkpoints16k_mssdcc-merge2-16r16-zzffko"
MODEL_PATH="path_to_saving_checkpoints16k_scale64_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
MODEL_PATH="path_to_saving_checkpoints16k_scale128_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
MODEL_PATH="path_to_saving_checkpoints16k_scale512_m64_w16s512_kojobntk_acc4_g4/checkpoint-1000"
# MODEL_PATH="path_to_saving_checkpoints32k_scale8_m64_w16s512_kojobntk_acc4_g4/checkpoint-600"
# MODEL_PATH="path_to_saving_checkpoints16k_scale2048_m64_w16s512_kojobntk_acc4_g4/checkpoint-1000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobntk_acc4_g4/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16384_m64_w16s512_kojobntk_acc4_g4/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints32k_scale4096_m64_w16s512_kojobntk_acc4_g2/checkpoint-1200"
MODEL_PATH="path_to_saving_checkpoints32k_z2off-m128-512r16-ko/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobPI_acc4_g4/checkpoint-2000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobntk_acc4_g4/checkpoint-3000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyntk_dyqek_m64_w16s512_zkojobPI128K_acc4_g4/checkpoint-500"
# MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyntk_dyqek_m64_w16s512_zkojobPI128K_acc4_g4/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyntk_dyqek_m64_w16s512_zkojobPI128K__acc4_g4/checkpoint-3500"
MODEL_PATH="path_to_saving_checkpoints16k_scale1632dypi_dyqek_m64_w16s512_zkojobPI128K_acc4_g4/checkpoint-500"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobntk_acc4_g4/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints32k_scale4096_m64_w16s512_zkojobntk_acc4_g2/checkpoint-2400"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyNTK_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4/checkpoint-4000"
MODEL_PATH="Mpath_to_saving_checkpoints16k_scale16_m64_w16s512_kojobNTK/checkpoint-1500"
MODEL_PATH="Mpath_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobNTK/checkpoint-4000"
MODEL_PATH="Mpath_to_saving_checkpoints16k_scale16_m64_w16s512_kojobNTK/checkpoint-5000"
# MODEL_PATH="Mpath_to_saving_checkpoints16k_scale64_m64_w16s512_kojobNTK/checkpoint-$STEPS"

# cd $MODEL_PATH
# python zero_to_fp32.py . pytorch_model.bin
# cd ..
# cd ..
# python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,o_proj,k_proj"
# CUDA_VISIBLE_DEVICES=1 python3 merge_lora_weights_and_save_hf_model.py \
#         --base_model /home/wangning/transformer-xl-master/output/llama \
#         --peft_model $MODEL_PATH \
#         --context_size 16384 \
#         --save_path /home/wangning/LongLoRA-main/$MODEL_PATH-7b-longlora-kvcc-merged
CUDA_VISIBLE_DEVICES=0,1,2,3 python passkey_retrivial_LBMs.py \
        --context_size 5000 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 320000 \
        --interval 100000 \
        --replace_l 9   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python passkey_retrivial_LBMs.py \
        --context_size 5000 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 320000 \
        --interval 70000 \
        --replace_l 9   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0,1,2,3 python passkey_retrivial_LBMs.py \
        --context_size 5000 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 320000 \
        --interval 120000 \
        --replace_l 9   \
        --replace_lm 0