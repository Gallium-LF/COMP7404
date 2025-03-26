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
