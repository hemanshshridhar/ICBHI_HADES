# Requirements

## Installation

To install the required dependencies, run the following commands:

```bash
pip install -r requirements.txt
conda install -c conda-forge compilers
conda install -c conda-forge ffmpeg=6
```

```bash
git clone https://github.com/MzeroMiko/VMamba.git
cd VMamba/kernels/selective_scan && pip install .
```
# Pre-trained Weights

To use the DASS pretrained models, download the checkpoints from the DASS release page and place them in the `pretrained_models` folder:

- **Download**: [DASS v0.2 pretrained checkpoints](https://github.com/Saurabhbhati/DASS/releases/tag/v0.2)
- **Save location**: create a folder named `pretrained_models` in this project root and put the downloaded checkpoint files inside it.

For example:

```bash
mkdir -p pretrained_models
# Move the downloaded .pth/.pt files into ./pretrained_models
mv /path/to/downloaded/checkpoint.pth ./pretrained_models/
```

# Usage

## For DASS
```
python main_v2.py --tag bs16_lr5e-5_ep50_seed2_ce_hades_stage2_blocks2,3 \
                  --seed 2 \
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
                  --model dass \
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
```

## References

The idea and code for this implementation is heavily inspired by the ICLR 2026 Paper "Graph Signal Processing Meets Mamba2: Adaptive Filter Bank via Delta Modulation"  (https://arxiv.org/abs/1234.12345) in which the Mixture fo experts is applied to "dts" parameter. 



## Plot Visualisation

Attention Map and Spectral Filter Response visulisation code is contained in visualize.ipynb (Attention Maps and Spectral filter Response)


# Results

<p align="center">
  <img src="/images/results.png" width="700"/>
  <br>
  <em>Result comaparison over 5 random seeds.</em>
</p>