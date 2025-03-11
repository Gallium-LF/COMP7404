import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compare_visually(output_img, gt_img, title="比较结果"):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(output_img, cmap='gray')
    plt.title("处理结果")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gt_img, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def evaluate_quality(output_img, gt_img):
    output_img = np.clip(output_img, 0, 1)
    gt_img = np.clip(gt_img, 0, 1)
    
    # 计算PSNR
    psnr = peak_signal_noise_ratio(gt_img, output_img, data_range=1.0)
    
    # 计算SSIM
    ssim = structural_similarity(gt_img, output_img, data_range=1.0)
    
    # 计算MSE
    mse = np.mean((gt_img - output_img) ** 2)
    
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MSE": mse
    }

def calculate_ipsnr(blurry_img, output_img, gt_img):
    blurry_img = np.clip(blurry_img, 0, 1)
    output_img = np.clip(output_img, 0, 1)
    gt_img = np.clip(gt_img, 0, 1)
    
    psnr_blurry = peak_signal_noise_ratio(gt_img, blurry_img, data_range=1.0)
    psnr_output = peak_signal_noise_ratio(gt_img, output_img, data_range=1.0)
    ipsnr = psnr_output - psnr_blurry
    
    return ipsnr

def save_comparison(blurry_img, output_img, gt_img, save_path):
    """比较图"""
    blurry_uint8 = (blurry_img * 255).astype(np.uint8)
    output_uint8 = (output_img * 255).astype(np.uint8)
    gt_uint8 = (gt_img * 255).astype(np.uint8)
    
    h, w = output_img.shape
    comparison = np.zeros((h, w*3), dtype=np.uint8)
    comparison[:, :w] = blurry_uint8
    comparison[:, w:2*w] = output_uint8
    comparison[:, 2*w:] = gt_uint8
    
    cv2.imwrite(str(save_path), comparison)

def evaluate_model(model, z_test, x_test, test_z_files, result_dir, border=13, safe_predict=None):
    """模型性能"""
    if safe_predict is None:
        safe_predict = lambda model, img: predict_image(model, img)
    
    # 保存目录
    comparison_dir = result_dir / "comparisons"
    comparison_dir.mkdir(exist_ok=True)
    
    all_metrics = []
    ipsnr_values = []
    for idx, (z_img, x_img, z_file) in enumerate(zip(z_test, x_test, test_z_files)):
        pred = safe_predict(model, z_img)
        common_h = min(pred.shape[0], x_img.shape[0])
        common_w = min(pred.shape[1], x_img.shape[1])
        if pred.shape != x_img.shape:
            pred_resized = pred[:common_h, :common_w]
            x_img_resized = x_img[:common_h, :common_w]
        else:
            pred_resized = pred
            x_img_resized = x_img
        print(f"评估尺寸: {pred_resized.shape}")
        
        try:
            # 计算指标
            metrics = evaluate_quality(pred_resized, x_img_resized)
            all_metrics.append(metrics)
            # 裁剪模糊图像
            z_img_resized = z_img[:min(common_h, z_img.shape[0]), :min(common_w, z_img.shape[1])]
            ipsnr = calculate_ipsnr(z_img_resized, pred_resized, x_img_resized)
            ipsnr_values.append(ipsnr)
            # 保存
            save_path = comparison_dir / f"comparison_{z_file.name}"
            save_comparison(z_img_resized, pred_resized, x_img_resized, save_path)
            # 输出指标
            print(f"图像 {idx+1}/{len(z_test)}: PSNR = {metrics['PSNR']:.2f}dB, SSIM = {metrics['SSIM']:.4f}, IPSNR = {ipsnr:.2f}dB")
        except Exception as e:
            print(f"处理图像 {idx+1} 时出错: {e}")
            continue
    
    # 计算平均指标
    if all_metrics:
        avg_metrics = {
            metric: np.mean([m[metric] for m in all_metrics]) 
            for metric in all_metrics[0]
        }
        avg_ipsnr = np.mean(ipsnr_values) if ipsnr_values else None
        
        # 输出平均指标
        print("\n平均指标:")
        print(f"平均 PSNR: {avg_metrics['PSNR']:.2f}dB")
        print(f"平均 SSIM: {avg_metrics['SSIM']:.4f}")
        print(f"平均 MSE: {avg_metrics['MSE']:.6f}")
        if avg_ipsnr is not None:
            print(f"平均 IPSNR: {avg_ipsnr:.2f}dB")
        
        return {
            "metrics": all_metrics,
            "avg_metrics": avg_metrics,
            "ipsnr": ipsnr_values,
            "avg_ipsnr": avg_ipsnr
        }
    else:
        print("没有成功评估任何图像")
        return None
