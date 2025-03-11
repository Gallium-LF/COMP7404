import torch
import numpy as np
from pathlib import Path
import cv2
from evaluate import evaluate_model
from mlp import DeblurMLP, load_image_pairs

def create_error_map(gt_img, pred_img):
    """创建误差图"""
    gt_img = np.clip(gt_img, 0, 1)
    pred_img = np.clip(pred_img, 0, 1)
    # 绝对误差
    error = np.abs(gt_img - pred_img)
    # 放大误差
    error_vis = error * 5 
    error_vis = np.clip(error_vis, 0, 1) 
    # uint8格式
    error_uint8 = (error_vis * 255).astype(np.uint8)
    return error_uint8

# 复用
BASE_DIR = r"/root/autodl-tmp/COMP7404-main/dataset"
Z_PATHS = {
    "train": Path(BASE_DIR) / "deblur_with_artifact/b/train",
    "val": Path(BASE_DIR) / "deblur_with_artifact/b/val",
    "test": Path(BASE_DIR) / "deblur_with_artifact/b/test"
}
X_PATHS = {
    "train": Path(BASE_DIR) / "GT/train",
    "val": Path(BASE_DIR) / "GT/val",
    "test": Path(BASE_DIR) / "GT/test"
}
result_dir = Path(r"/root/autodl-tmp/COMP7404-main")
model_path = result_dir / "deblur_mlp_b.pth"

def safe_predict(model, z_image):
    try:
        return predict_image(model, z_image)
    except torch.cuda.OutOfMemoryError:
        print("GPU内存不足，尝试使用CPU推理...")
        model = model.cpu()
        return predict_image(model, z_image)

def predict_image(model, z_image):
    model.eval()
    h, w = z_image.shape
    
    # 使用镜像填充处理边界
    pad_size = 20  
    z_padded = np.pad(z_image, pad_size, mode='reflect')
    
    output = np.zeros_like(z_padded)
    counts = np.zeros_like(z_padded)
    
    # 在填充后的图像上滑动窗口
    padded_h, padded_w = z_padded.shape
    for y in range(0, padded_h-39+1, 3):
        for x in range(0, padded_w-39+1, 3):
            # 提取输入块
            z_patch = z_padded[y:y+39, x:x+39]
            z_tensor = torch.FloatTensor(z_patch).reshape(-1).unsqueeze(0)
            z_tensor = z_tensor * 2 - 1  # 归一化
            
            # GPU加速
            if torch.cuda.is_available():
                z_tensor = z_tensor.cuda()
            
            # 预测
            with torch.no_grad():
                x_tensor = model(z_tensor).cpu().numpy().squeeze()
                x_patch = (x_tensor.reshape(13, 13) + 1) / 2  # 反归一化
            
            # 将预测结果放回，中心对齐
            output[y+13:y+26, x+13:x+26] += x_patch
            counts[y+13:y+26, x+13:x+26] += 1
    
    # 平均重叠区域
    valid_mask = counts > 0
    output[valid_mask] /= counts[valid_mask]
    
    final_output = output[pad_size:pad_size+h, pad_size:pad_size+w]
    
    return np.clip(final_output, 0, 1)
    
    # 裁剪图像边缘，移除黑边
    #border = 13  # 裁剪边界宽度
    #final_cropped = final[border:h-border, border:w-border]
    #final = final[pad:pad+z_image.shape[0], pad:pad+z_image.shape[1]]
    #return np.clip(final, 0, 1)

if __name__ == '__main__':
    print("加载数据...")
    z_test, x_test = load_image_pairs(Z_PATHS["test"], X_PATHS["test"])
    test_z_dir = Z_PATHS["test"]
    test_z_files = sorted(test_z_dir.glob("*.*"))
    
    #测试用
    max_images = 3
    z_test = z_test[:max_images]
    x_test = x_test[:max_images]
    test_z_files = test_z_files[:max_images]
    print(f"前 {max_images} 张图像...")
    
    print("加载模型...")
    model = DeblurMLP().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    print("开始评估...")
    evaluate_model(
        model=model,
        z_test=z_test,
        x_test=x_test,
        test_z_files=test_z_files,
        result_dir=result_dir,
        border=13,
        safe_predict=safe_predict
    )
    
    print("\n生成误差图...")
    error_dir = result_dir / "error_maps"
    error_dir.mkdir(exist_ok=True)
    
    # 生成误差图
    for idx, (z_img, x_img, z_file) in enumerate(zip(z_test, x_test, test_z_files)):
        # 预测结果
        pred = safe_predict(model, z_img)
        
        # 裁剪GT
        h, w = pred.shape
        h_orig, w_orig = z_img.shape
        
        #x_img_cropped = x_img[border:min(border+h, h_orig), border:min(border+w, w_orig)]
        common_h = min(pred.shape[0], x_img.shape[0])
        common_w = min(pred.shape[1], x_img.shape[1])
        x_img_cropped = x_img[:common_h, :common_w]
        pred_cropped = pred[:common_h, :common_w]
        
        # 调整尺寸
        if pred.shape != x_img_cropped.shape:
            min_h = min(pred.shape[0], x_img_cropped.shape[0])
            min_w = min(pred.shape[1], x_img_cropped.shape[1])
            pred = pred[:min_h, :min_w]
            x_img_cropped = x_img_cropped[:min_h, :min_w]
        
        error_map = create_error_map(x_img_cropped, pred)
        error_path = error_dir / f"error_{z_file.name}"
        cv2.imwrite(str(error_path), error_map)
        
        # 对比图
        h, w = pred.shape
        comparison = np.zeros((h, w*3), dtype=np.uint8)
        comparison[:, :w] = (x_img_cropped * 255).astype(np.uint8)  
        comparison[:, w:2*w] = (pred * 255).astype(np.uint8)        
        comparison[:, 2*w:] = error_map                            
        
        comparison_path = result_dir / "error_comparisons" / f"comp_{z_file.name}"
        comparison_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(comparison_path), comparison)
        
        print(f"已生成误差图 {idx+1}/{len(z_test)}: {error_path}")
        
    print("评估完成！")