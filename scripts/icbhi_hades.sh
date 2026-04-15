#!/bin/bash

# Configuration
MODEL="dass"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do

        TAG="bs16_lr5e-5_ep50_seed${s}_ce_hades_stage2_blocks2,3"

        CUDA_VISIBLE_DEVICES=0 python main_v2.py --tag $TAG \
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
                                        --method ce \
                                        --hades_on \
                                        --hades_stage 2 \
                                        --hades_blocks 2,3 \
                                        --alpha_lb 1.0 \
                                        --alpha_div 1.0

        # For evaluation, use:
        # --eval \
        # --pretrained \
        # --pretrained_ckpt ./save/icbhi_${m}_ce_bs16_lr5e-5_ep50_seed${s}/best.pth

    done
done