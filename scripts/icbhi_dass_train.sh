#!/bin/bash

# Configuration
MODEL="dass"
SEED="1 2 3 4 5"

for s in $SEED
do
    for m in $MODEL
    do
        TAG="bs8_lr5e-5_ep50_seed${s}"
        CUDA_VISIBLE_DEVICES=0 python main.py --tag $TAG \
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
                                        --method ce

                                        # only for evaluation, add the following arguments
                                        # --eval \
                                        # --pretrained \
                                        # --pretrained_ckpt ./save/icbhi_${m}_ce_bs8_lr5e-5_ep50_seed${s}/best.pth

    done
done

