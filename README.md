# COMP7404
This is a class project about image deconvolution

teammate: CHENG Qirui, HUANG Zhenbang, LIN Ziyang, ZHANG Yushan 

## Preprocessing Steps

Based on the BSDS500 dataset, the images are first converted to grayscale and resized to 512×512. Then, they are further adjusted according to different experimental parameter settings as follows:

## Experimental Settings

- **(a) Gaussian Blur (σ = 1.6) and AWG Noise (σ = 0.04)**  
  - A Gaussian blur with a standard deviation of **1.6** is applied using a **25 × 25** kernel.  
  - Additive white Gaussian (AWG) noise with **σ = 0.04** is introduced.

- **(b) Gaussian Blur (σ = 1.6) and AWG Noise (σ = 2/255 ≈ 0.008)**  
  - A Gaussian blur with a standard deviation of **1.6** is applied using a **25 × 25** kernel.  
  - AWG noise with **σ = 2/255 (≈ 0.008)** is introduced.

- **(c) Gaussian Blur (σ = 3.0) and AWG Noise (σ = 0.04)**  
  - A Gaussian blur with a standard deviation of **3.0** is applied using a **25 × 25** kernel.  
  - AWG noise with **σ = 0.04** is introduced.

- **(d) Square Blur (Box Blur) and AWG Noise (σ = 0.01)**  
  - A square (box) blur with a **19 × 19** kernel is applied.  
  - AWG noise with **σ = 0.01** is introduced.

- **(e) Motion Blur and AWG Noise (σ = 0.01)**  
  - Motion blur is applied based on the method described in **[21]**.  
  - AWG noise with **σ = 0.01** is introduced.

## Direct Deblurring

We implemented two distinct methods for image deblurring: **Gradient Descent** and **Fourier-based deblurring**. While both approaches successfully restored sharpness to the blurred images, they also introduced certain artifacts that required attention in the second stage of processing.

### Method Comparison

- `deblur_gd.py`: Implements the **gradient descent** method.
- `deblur_fl.py`: Implements the **Fourier-based** method.

After evaluating the results, our group chose the **gradient descent method** due to its overall better performance.

### Batch Processing

The script `deblur_final.py` was used to apply the gradient descent-based method to a batch of images.  
All output images—including those containing deblurring artifacts—are stored in the `deblur_with_artifact` folder.

# MLP

Image deblurring is achieved based on a multi-layer perceptron (MLP), including the entire process of training, reasoning, and evaluation.

## Data structure

```
.
├── dataset/                    
│   ├── GT/                     
│   └── deblur_with_artifact/   
├── mlp.py                                     
├── evaluate.py                 
├── run.py                     
├── all_run.sh                  
└── results/                    
    ├── test_results/           
    ├── error_maps/             
    └── comparisons/            
```

## Environmental requirements

- Python 3.8+
- Key dependencies:

```bash
pip install torch==1.13.1 opencv-python==4.7.0.68 scikit-image==0.19.3 tqdm matplotlib
```

- GPU support: CUDA 11.6+

## Quick start

### 1. Data preparation

The data set is organized as follows：

```
dataset/
├── GT/
│   ├── train/
│   ├── val/
│   └── test/
└── deblur_with_artifact/
    └── [a|b|c|d|e]/  # Different fuzzy groups
        ├── train/
        ├── val/
        └── test/
```

### 2. Model training

Individual model training (please note to change the run directory)

```bash
python mlp.py
```

Batch execution (please note to change the file name in all_run.sh) :

```bash
chmod +x all_run.sh
nohup ./all_run.sh > training.log 2>&1 & 
tail -f run.log  
```

### 3. Evaluate

Please note to change the directory name.

```bash
python run.py 
```

## Technical details

### Model architecture

```python
DeblurMLP(
  (mlp): Sequential(
    (0): Linear(1521→2047)
    (1): Tanh()
    (2-5): 3x[Linear(2047→2047) + Tanh()]
    (6): Linear(2047→169)
  )
)
```

### Training configuration

- Optimizer: SGD with Momentum (β=0.9)
- Learning rate: 0.01 (step attenuation)
- Regularization: L2 weight decay (λ=1e-4)
- Batch size: 128
- Training cycle: 100

### Data preprocessing

1. Image segmentation: 39x39 input → 13x13 output
2. Slide window: Step size 3 pixels
3. Normalization: Linear scaling to [-1,1]
4. Border processing: image fill 20 pixels

## Result output

### Output file

| file type | format | example |
|---------|------|-----|
| deblurring result | PNG | 'test_results/100001.png' |
| error heat map | PNG | 'error_maps/100001.png' |
| comparisons | PNG| 'comparisons /100001.png' |
