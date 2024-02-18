import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import zipfile
import os
import numpy as np

class ImageNet64(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = os.listdir(data_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.data_dir, self.image_files[idx])
        with open(image_file, 'rb') as f:
            image = np.fromfile(f, dtype=np.uint8)
        image = image.reshape(64, 64, 3)  # 假设图像是 RGB 格式
        if self.transform:
            image = self.transform(image)
        return image
    
# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

test_dataset = ImageNet64(data_dir=r'E:\datasets\Imagenet64_val', transform=transform)