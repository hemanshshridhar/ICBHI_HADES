#!/bin/bash

# Configuration
MODEL="dass"
SEED="1 2"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs16_lr5e-5_ep50_seed${s}_dual_patchmix_cl_gaussian_temp0.2_kernel5_sigma3_blur2_blurblocks2,3"
        CUDA_VISIBLE_DEVICES=7 python main_v2.py --tag $TAG \
                                        --seed $s \
                                        --dataset icbhi \
                                        --class_split lungsound \
                                        --n_cls 4 \
                                        --epochs 50 \
                                        --batch_size 16 \
                                        --optimizer adam \
                                        --learning_rate 5e-5 \
                                        --weight_decay 1e-6 \
                                        --cosine \
                                        --desired_length 8 \
                                        --model $m \
                                        --test_fold official \
                                        --pad_types repeat \
                                        --resz 1 \
                                        --n_mels 128 \
                                        --ma_update \
                                        --ma_beta 0.5 \
                                        --pretrained \
                                        --from_sl_official \
                                        --audioset_pretrained \
                                        --method patchmix_cl \
                                        --cl_mode dual \
                                        --gaussian_blur \
                                        --temperature 0.2 \
                                        --proj_dim 768 \
                                        --alpha 1.0 \
                                        --mix_beta 1.0 \
                                        --kernel_size 5 \
                                        --sigma 3.0 \
                                        --blur_stage 2 \
                                        --blur_blocks 2,3

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_${m}_ce_bs8_lr5e-5_ep50_seed${s}/best.pth

    done
done

