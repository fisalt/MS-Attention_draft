#!/bin/bash
# MODEL_PATH="path_to_saving_checkpoints32k_z2off-m128-512r16-ko/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16kppl_m64-512r16-ko/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojob/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16_m64_w16s512_kojobntk_acc4_g4/checkpoint-3000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale1_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
MODEL_PATH="path_to_saving_checkpoints16k_scale64_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk_acc4_g4/checkpoint-3500"
# MODEL_PATH="path_to_saving_checkpoints16k_mssdcc-merge2-16r16-zzffko"
# MODEL_PATH="path_to_saving_checkpoints16k_scale128_m64_w16s512_kojobntk_acc4_g4/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobPI_acc4_g4/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobntk_acc4_g4/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyntk_dyqek_m64_w16s512_zkojobPI128K_acc4_g4/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyntk_dyqek_m64_w16s512_zkojobPI128K__acc4_g4/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobntk_acc4_g4/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyNTK_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4/checkpoint-4000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyNTK_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyPI_dyqek_m64_w16s512_zkojobPI32K_acc4_g4/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale32nodyPI_dyqek_m64_w16s512_zkojobPI32K_acc4_g4/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale8nodyPI_dyqek_m64_w16s512_zkojobPI32K_acc4_g4/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_bypart/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_zeropart/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod8K/checkpoint-2000"
# MODEL_PATH="path_to_saving_checkpoints32k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_ModZ16KS8K/checkpoint-2500"
# MODEL_PATH="path_to_saving_checkpoints32k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod16K/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints32k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32K/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1M/checkpoint-4000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MinfiC/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints32k_scale4096nodyNTKdim_dyqek_m128_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod64K/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MZ4/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MinfiC1e-1/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32Kpmaxs32Ke1MinfiC5e-1/checkpoint-500"
MODEL_PATH="path_to_saving_checkpoints32k_scale4096nodyNTKdim_dyqek_m128_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_ModDynamicS32Ka16K/checkpoint-4000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_ModDynamicS8Ka16K/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints32k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_Mod32KC1e-1/checkpoint-5000"
MODEL_PATH="path_to_saving_checkpoints32k_scale1nodyNTKdim_dyqek_m128_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_ModDynamicS32Ka16K/checkpoint-3500"

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

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LB.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 58008 \
        --interval 8000 \
        --replace_l 8   \
        --scaling_factor 0.5   \
        --num_tests 2   \
        --replace_lm 0   \
        --dynamic 1 

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LB.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 58008 \
        --interval 8000 \
        --replace_l 8   \
        --scaling_factor 0.25   \
        --num_tests 2   \
        --replace_lm 0   \
        --dynamic 1  

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LB.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 58008 \
        --interval 8000 \
        --replace_l 8   \
        --scaling_factor 0.1   \
        --num_tests 2   \
        --replace_lm 0   \
        --dynamic 1  
