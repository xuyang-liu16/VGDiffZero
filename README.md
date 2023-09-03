# VGDiffZero: Text-to-image Diffusion Models Can Be Zero-shot Visual Grounders
Official PyTorch implementation of our paper <br>
**VGDiffZero: Text-to-image Diffusion Models Can Be Zero-shot Visual Grounders** <br>
Xuyang Liu, Siteng Huang, Yachen Kang, Honggang chen, and Donglin Wang <br>
_Preprint, Sep 2023_ <br>

## Overview
<p align="center"> <img src=
Given an input image, isolated proposals are generated via cropping and masking, and then encoded individually into latent vectors $Z_0$. Gaussian noise $\epsilon$ sampled from $\mathcal{N}(0, 1)$ is injected into each latent vector to obtain noised latent representations $Z_\text{noised}$. Subsequently, each noised latent together with the text embeddings is fed into the UNet to select the best matching proposal as the final prediction.

## Code

### Installation 
Create a conda environment and activate with the following command:
```bash
conda env create -f environment.yml
conda activate VGDiffZero
```
