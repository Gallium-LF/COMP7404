# require opencv-python-headless, numpy, matplotlib, scipy, tqdm
# this file is used to test the gradient descent method of deblurring
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os
import tqdm
import matplotlib.pyplot as plt

def gaussian_kernel(size, sigma):
    k = cv2.getGaussianKernel(size, sigma)
    kernel = k @ k.T
    return kernel

def box_blur(size=19):
    kernel = np.ones((size, size), dtype=np.float32) / (size**2)
    return kernel
    
def motion_blur(kernel_size=19):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    return kernel

def compute_gradient(x):
    grad_x = np.roll(x, -1, axis=1) - x
    grad_y = np.roll(x, -1, axis=0) - x
    return grad_x, grad_y

def inverse_filtering_gradient_descent(y, psf, alpha=0.01, beta=0.001, num_iters=100, lr=0.1):
    x = np.copy(y)
    psf_flip = np.flipud(np.fliplr(psf))

    for i in range(num_iters):
        residual = y - scipy.signal.convolve2d(x, psf, mode='same', boundary='symm')

        grad_x, grad_y = compute_gradient(x)
        grad_term = np.roll(grad_x, 1, axis=1) - grad_x + np.roll(grad_y, 1, axis=0) - grad_y

        grad = -scipy.signal.convolve2d(residual, psf_flip, mode='same', boundary='symm') + alpha * grad_term + beta * x

        x = x.astype(np.float32)
        x -= lr * grad

        x = np.clip(x, 0, 255).astype(np.uint8)

    return x


input_dir = "dataset/blurred/c/train"
output_dir = "dataset/deblur_with_artifact/c/train"


os.makedirs(output_dir, exist_ok=True)

psf = gaussian_kernel(25, 3)

for filename in tqdm.tqdm(os.listdir(input_dir)):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    blurry_noisy_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if blurry_noisy_img is None:
        print(f"Error: Failed to load {input_path}")
        continue

    restored_img = inverse_filtering_gradient_descent(blurry_noisy_img, psf, alpha=0.01, beta=0, num_iters=100, lr=0.1)

    cv2.imwrite(output_path, restored_img)
    print(f"Processed: {output_path}")



blurry_noisy_img = cv2.imread("dataset\\blurred\\d\\train\\2092.jpg", cv2.IMREAD_GRAYSCALE)
psf = box_blur(19) 

restored_img = inverse_filtering_gradient_descent(blurry_noisy_img, psf, alpha=0.02, beta=0.001, num_iters=100, lr=0.05)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Blurred + Noisy Image")
plt.imshow(blurry_noisy_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Restored Image (Gradient Descent)")
plt.imshow(restored_img, cmap="gray")
plt.show()

cv2.imwrite("restored_image_gradient_descent.jpg", restored_img)