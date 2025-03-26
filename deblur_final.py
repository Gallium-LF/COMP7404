import os
import cv2
import torch
import numpy as np
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def gaussian_blur(size, sigma):
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

def inverse_filtering_cuda(y, psf, alpha=0.01, beta=0, num_iters=100, lr=0.1):
    y = torch.tensor(y, dtype=torch.float32, device=device) / 255.0
    x = y.clone().requires_grad_(True)
    psf = torch.tensor(psf, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) + 1e-6
    optimizer = torch.optim.Adam([x], lr=lr)
    psf_flip = torch.flip(psf, [2, 3])

    for _ in range(num_iters):
        optimizer.zero_grad()
        blurred_x = F.conv2d(F.pad(x.unsqueeze(0).unsqueeze(0), (psf.shape[-1]//2,)*4, mode='reflect'), psf)
        residual = y - blurred_x.squeeze()

        grad_x = x[:, 1:] - x[:, :-1]
        grad_y = x[1:, :] - x[:-1, :]
        loss = torch.norm(residual) ** 2 + alpha * (torch.norm(grad_x) ** 2 + torch.norm(grad_y) ** 2) + beta * torch.norm(x) ** 2

        loss.backward()
        optimizer.step()
        x.data = torch.clamp(x.data, 0, 1)

    return (x.detach().cpu().numpy() * 255).astype(np.uint8)


blurry_noisy_img = cv2.imread("dataset\\blurred\\c\\val\\3096.jpg", cv2.IMREAD_GRAYSCALE)
print(blurry_noisy_img.shape)
psf = gaussian_blur(25,3)

restored_img = inverse_filtering_cuda(blurry_noisy_img, psf, alpha=0.1, beta=0, num_iters=100, lr=0.1)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Blurred + Noisy Image")
plt.imshow(blurry_noisy_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("Restored Image (Gradient Descent)")
plt.imshow(restored_img, cmap="gray")
plt.show()



input_dir = "dataset/blurred/e/test"
output_dir = "dataset/deblur_with_artifact/e/test"
os.makedirs(output_dir, exist_ok=True)

# choose the PSF
#psf = gaussian_blur(25, 3)
#psf = box_blur(19)
psf = motion_blur(19)

for filename in tqdm.tqdm(os.listdir(input_dir)):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    blurry_noisy_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if blurry_noisy_img is None:
        print(f"Error: Failed to load {input_path}")
        continue

    restored_img = inverse_filtering_cuda(blurry_noisy_img, psf, alpha=0.01, beta=0, num_iters=100, lr=0.1)

    cv2.imwrite(output_path, restored_img)
    print(f"Processed: {output_path}")

# Compare the restored image with the ground truth
gt_img = cv2.imread("dataset\\GT\\val\\3096.jpg", cv2.IMREAD_GRAYSCALE)
restored_img = cv2.imread("dataset\\deblur_with_artifact\\c\\val\\3096.jpg", cv2.IMREAD_GRAYSCALE)

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