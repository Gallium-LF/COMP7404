import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)


def process_images(input_folder, output_folder, size=(512, 512)):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*.*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"无法读取图像: {img_file}")
                continue

            img_resized = cv2.resize(img, size)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), img_gray)
            print(f"处理完成: {output_file}")


def gaussian_filter(image, size=25, sigma=1.6):
    gaussian_1d = cv2.getGaussianKernel(size, sigma)
    gaussian_2d = gaussian_1d @ gaussian_1d.T
    blurred_image = cv2.filter2D(image, -1, gaussian_2d)
    return blurred_image


def apply_box_blur(image, size=19):
    box_blur = np.ones((size, size), dtype=np.float32) / (size**2)
    return cv2.filter2D(image, -1, box_blur)


def apply_motion_blur(image, kernel_size=19):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
    return cv2.filter2D(image, -1, kernel)


def add_gaussian_noise(image, std=0.04):
    row, col = image.shape
    gauss = np.random.normal(0, std, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def process_images_blurred(kernel_size, sigma, std, input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*.*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像: {img_file}")
                continue

            img_gray = gaussian_filter(img, size=kernel_size, sigma=sigma)
            img_gray = add_gaussian_noise(img_gray, std=std)

            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), img_gray)
            print(f"处理完成: {output_file}")


def process_images_box_blurred(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*.*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像: {img_file}")
                continue

            img_gray = apply_box_blur(img, size=19)
            img_gray = add_gaussian_noise(img_gray, std=0.01)

            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), img_gray)
            print(f"处理完成: {output_file}")


def process_images_motion_blurred(input_folder, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for img_file in input_path.glob("*.*"):
        if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像: {img_file}")
                continue

            img_gray = apply_motion_blur(img, kernel_size=19)
            img_gray = add_gaussian_noise(img_gray, std=0.01)

            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), img_gray)
            print(f"处理完成: {output_file}")
