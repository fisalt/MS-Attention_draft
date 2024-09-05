#!/bin/bash
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojob/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk/checkpoint-5000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale16_m64_w16s512_kojobntk_acc4_g4/checkpoint-3000"
# MODEL_PATH="path_to_saving_checkpoints16k_scale1_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
# MODEL_PATH="path_to_saving_checkpoints16k_scale64_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
MODEL_PATH="path_to_saving_checkpoints16k_glolen_m64_w16s512_kojobntk_acc4_g4/checkpoint-3500"
MODEL_PATH="path_to_saving_checkpoints16k_scale128_m64_w16s512_kojobntk_acc4_g4/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale512_m64_w16s512_kojobntk_acc4_g4/checkpoint-1000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096_m64_w16s512_kojobntk_acc4_g4/checkpoint-2000"
MODEL_PATH="path_to_saving_checkpoints16k_scale16384_m64_w16s512_kojobntk_acc4_g4/checkpoint-1500"
MODEL_PATH="path_to_saving_checkpoints32k_scale4096_m64_w16s512_kojobntk_acc4_g2/checkpoint-1200"
MODEL_PATH="path_to_saving_checkpoints16k_scale16384_m64_w16s512_zkojobntk_acc4_g4/checkpoint-500"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobntk_acc4_g4/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints16k_scale4096nodyntk_dyqek_m64_w16s512_zkojobntk_acc4_g4/checkpoint-3000"
MODEL_PATH="path_to_saving_checkpoints32k_scale16nodyNTKdim_dyqek_m64_w16s512_zkojobNTK32K_acc4_g4_lora64_noseqlen_ModZ16KS8K/checkpoint-2500"

# MODEL_PATH="path_to_saving_checkpoints32k_scale8_m64_w16s512_kojobntk_acc4_g4/checkpoint-600"
# cd $MODEL_PATH
# python zero_to_fp32.py . pytorch_model.bin
# cd ..
# cd ..
# python3 get_trainable_weights.py --checkpoint_path $MODEL_PATH --trainable_params "embed,norm,patchscale,o_proj,k_proj"
# CUDA_VISIBLE_DEVICES=0 python3 merge_lora_weights_and_save_hf_model.py \
#         --base_model /home/wangning/transformer-xl-master/output/llama \
#         --peft_model $MODEL_PATH \
#         --context_size 32768 \
#         --save_path /home/wangning/LongLoRA-main/$MODEL_PATH-7b-longlora-kvcc-merged
# CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBF4c.py \
#         --context_size 65536 \
#         --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
#         --max_tokens 594700 \
#         --interval 500000 \
#         --scaling_factor 4096   \
#         --num_tests 2   \
#         --replace_l 8   \
#         --replace_lm 0

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBF4c.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 594700 \
        --interval 500000 \
        --scaling_factor 512   \
        --num_tests 2   \
        --replace_l 8   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBF4c.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 594700 \
        --interval 500000 \
        --scaling_factor 2048   \
        --num_tests 2   \
        --replace_l 8   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBF4c.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 594700 \
        --interval 500000 \
        --scaling_factor 16384   \
        --num_tests 2   \
        --replace_l 8   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBF4c.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 594700 \
        --interval 500000 \
        --scaling_factor 8192   \
        --num_tests 2   \
        --replace_l 8   \
        --replace_lm 0

CUDA_VISIBLE_DEVICES=0 python3 passkey_retrivial_LBF4c.py \
        --context_size 65536 \
        --base_model $MODEL_PATH-7b-longlora-kvcc-merged \
        --max_tokens 594700 \
        --interval 500000 \
        --scaling_factor 2048   \
        --num_tests 2   \
        --replace_l 8   \
        --replace_lm 0