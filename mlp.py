import os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# 基础路径
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

class DeblurDataset(Dataset):
    def __init__(self, z_images, x_images):
        self.z_images = z_images
        self.x_images = x_images
        self.indices = []
        
        # 预计算块索引
        for img_idx in range(len(z_images)):
            z_img = z_images[img_idx]
            x_img = x_images[img_idx]
            h, w = z_img.shape
            for y in range(0, h - 39 + 1, 3):  # 步长3
                for x in range(0, w - 39 + 1, 3):
                    self.indices.append((img_idx, y, x))
                    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_idx, y, x = self.indices[idx]
        z_patch = self.z_images[img_idx][y:y+39, x:x+39]
        x_patch = self.x_images[img_idx][y+13:y+26, x+13:x+26]
        
        z_tensor = torch.FloatTensor(z_patch).reshape(-1) * 2 - 1
        x_tensor = torch.FloatTensor(x_patch).reshape(-1) * 2 - 1
        return z_tensor, x_tensor

def load_image_pairs(z_dir, x_dir):
    """加载图像对"""
    z_images, x_images = [], []
    z_files = sorted(z_dir.glob("*.*"))
    x_files = sorted(x_dir.glob("*.*"))
    
    for z_file, x_file in zip(z_files, x_files):
        z_img = cv2.imread(str(z_file), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        x_img = cv2.imread(str(x_file), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if z_img.shape != x_img.shape:
            raise ValueError(f"图像尺寸不匹配: {z_file.name} {z_img.shape} vs {x_file.name} {x_img.shape}")
        z_images.append(z_img)
        x_images.append(x_img)
    return z_images, x_images

class DeblurMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(39*39, 2047),#3.2
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(2047, 2047), nn.Tanh()) for _ in range(3)],
            nn.Linear(2047, 13*13)
        )
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.mlp(x)

def train_mlp(z_images, x_images):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"训练设备: {device}")
    
    dataset = DeblurDataset(z_images, x_images)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=16, 
        pin_memory=True,
    )
    
    model = DeblurMLP().to(device)
    #SGD优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # 进度条
    for epoch in tqdm(range(100), desc="训练进度"):
        model.train()
        total_loss = 0.0
        for z_batch, x_batch in loader:
            z_batch = z_batch.to(device, non_blocking=True)
            x_batch = x_batch.to(device, non_blocking=True)
            
            pred = model(z_batch)
            loss = criterion(pred, x_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * z_batch.size(0)
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f}")
    
    return model

def predict_image(model, z_image):
    model.eval()
    h, w = z_image.shape
    
    # 使用镜像填充处理边界
    pad_size = 20  # 略大于13
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
    
    # 裁剪回原始大小
    final_output = output[pad_size:pad_size+h, pad_size:pad_size+w]
    
    return np.clip(final_output, 0, 1)

# pred = predict_image(model, z_img)
# pred_uint8 = (pred * 255).astype(np.uint8)
# save_path = test_save_dir / z_file.name
# cv2.imwrite(str(save_path), pred_uint8)

if __name__ == '__main__':
    # 加载数据
    try:
        print("开始加载数据...")
        z_train, x_train = load_image_pairs(Z_PATHS["train"], X_PATHS["train"])
        z_test, x_test = load_image_pairs(Z_PATHS["test"], X_PATHS["test"])
        z_val, x_val = load_image_pairs(Z_PATHS["val"], X_PATHS["val"])
        print(f"训练集: {len(z_train)}张, 测试集: {len(z_test)}张, 验证集: {len(z_val)}张")
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        exit(1)
    
    result_dir = Path(r"/root/autodl-tmp/COMP7404-main/")
    result_dir.mkdir(exist_ok=True)
    model_path = result_dir / "deblur_mlp_b.pth"
     
    if not model_path.exists():
        print("开始训练新模型...")
        model = train_mlp(z_train, x_train)
        torch.save(model.state_dict(), model_path)
    else:
        print("加载已有模型...")
        model = DeblurMLP().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    def safe_predict(model, z_image):
        try:
            return predict_image(model, z_image)
        except torch.cuda.OutOfMemoryError:
            print("GPU内存不足，尝试使用CPU推理...")
            model = model.cpu()
            return predict_image(model, z_image)
    
    # ============== 测试集推理 ==============
    print("\n开始推理测试集...")
    test_save_dir = result_dir / "test_results1"
    test_save_dir.mkdir(exist_ok=True)
    
    # 获取测试集文件名列表
    test_z_dir = Z_PATHS["test"]
    test_z_files = sorted(test_z_dir.glob("*.*"))
    
    # 测试用
    #PROCESS_NUM = 3 
    #_test = z_test[:PROCESS_NUM]               
    #test_z_files = test_z_files[:PROCESS_NUM]  

    
    # 推理并保存
    for idx, (z_img, z_file) in enumerate(zip(z_test, test_z_files)):
        try:
            # 推理
            pred = safe_predict(model, z_img)
            
            # 保存
            pred_uint8 = (pred * 255).astype(np.uint8)
            save_path = test_save_dir / z_file.name
            cv2.imwrite(str(save_path), pred_uint8)
            print(f"({idx+1}/{len(z_test)}) 已保存测试结果: {save_path}")
        except Exception as e:
            print(f"处理 {z_file.name} 失败: {str(e)}")
    # ==============================================
    
    print("处理完成！")
