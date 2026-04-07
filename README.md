# Requirements

## Installation

To install the required dependencies, run the following commands:

```bash
pip install -r requirements.txt
conda install -c conda-forge compilers
conda install -c conda-forge ffmpeg=6
```

```bash
git clone https://github.com/Saurabhbhati/DASS.git
cd DASS/kernels/selective_scan && pip install --no-build-isolation .
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

# Usage (Added Gaussian Convolution hyperparameters in the CLI args)

## For DASS fine-tuning
```
sh ./scripts/icbhi_dass_train.sh
```


## For Lung-SRAD
```
sh ./scripts/icbhi_dual_patchmix_gaussian.sh
```


## Plot Visualisation
Attention Map and Spectral Filter Response visulisation code is contained in visualize.ipynb (Attention Maps and Spectral filter Response)


# Results

<p align="center">
  <img src="/images/results.png" width="700"/>
  <br>
  <em>Result comaparison over 5 random seeds.</em>
</p>
