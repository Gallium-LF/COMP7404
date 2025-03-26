# require opencv-python-headless, numpy, matplotlib, scipy, tqdm
# this file is used to test the fluourier method of deblurring
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import os
import tqdm

def gaussian_kernel(size, sigma):
    # Create a 1D Gaussian kernel
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

def center_psf(psf, shape):
    # Center the PSF
    # Reason: The PSF is usually defined in the center of the image
    psf_padded = np.zeros(shape, dtype=np.float32)
    h, w = psf.shape
    ph, pw = (shape[0] - h) // 2, (shape[1] - w) // 2
    psf_padded[ph:ph+h, pw:pw+w] = psf
    return np.fft.ifftshift(psf_padded)

def inverse_filtering(y, psf, alpha=0.01, beta=0.01):
    """
    Inverse filtering
    :param y: Blurred image
    :param psf: Point spread function
    :param alpha: Regularization parameter
    :param beta: Regularization parameter
    :return: Deblurred image
    """
    original_shape = y.shape

    psf = center_psf(psf, y.shape)

    Y = np.fft.fft2(y)
    V = np.fft.fft2(psf)
    V_conj = np.conj(V)

    G = np.abs(np.fft.fft2(np.array([[1, -1], [-1, 1]]), s=y.shape)) ** 2
    R = V_conj / (np.abs(V) ** 2 + alpha * G + beta)

    X_est = np.fft.ifft2(R * Y).real

    X_est = X_est[:original_shape[0], :original_shape[1]]  

    return np.clip(X_est, 0, 255).astype(np.uint8)


blurry_noisy_img = cv2.imread("dataset\\blurred\\c\\val\\3096.jpg", cv2.IMREAD_GRAYSCALE)

psf = gaussian_kernel(25, 1.6)

restored_img = inverse_filtering(blurry_noisy_img, psf)

plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title("Blurred + Noisy Image")
plt.imshow(blurry_noisy_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Restored Image")
plt.imshow(restored_img, cmap="gray")
plt.show()

cv2.imwrite("restored_image.png", restored_img)


gt_img = cv2.imread("dataset\\GT\\val\\3096.jpg", cv2.IMREAD_GRAYSCALE)
diff_img = cv2.subtract(gt_img, restored_img)
print("PSNR:", cv2.PSNR(gt_img, restored_img))
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Ground Truth")
plt.imshow(gt_img, cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Restored Image")
plt.imshow(restored_img, cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Difference")
plt.imshow(diff_img, cmap="gray")
plt.show()


diff_img = cv2.subtract(gt_img, restored_img)

print("PSNR:", cv2.PSNR(gt_img, restored_img))

plt.figure(figsize=(5, 5))
plt.imshow(gt_img, cmap="gray")
plt.axis('off')
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(restored_img, cmap="gray")
plt.axis('off')
plt.show()

plt.figure(figsize=(5, 5))
plt.imshow(diff_img, cmap="gray")
plt.axis('off')
plt.show()