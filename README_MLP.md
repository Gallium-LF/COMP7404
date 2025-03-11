# MLP

基于多层感知机（MLP）实现图像去模糊，包含训练、推理与评估全流程。

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