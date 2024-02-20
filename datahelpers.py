import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CocoDetection
import os
import config
import numpy as np

class ImageNet64(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = self.get_img_list()
        self.labels = [int(x.strip()) for x in open(data_dir + '/labels.txt', 'r').readlines() if x.strip()]

    def get_img_list(self):
        li = [x for x in os.listdir(self.data_dir) if x != 'labels.txt']
        li = sorted(li, key=lambda x:int(x.split('.')[0]))      # Use file number to sort file list.
        return li

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = os.path.join(self.data_dir, self.image_files[idx])
        image = np.load(image_file) # Load image. Shape: (3*64*64). Already normalized.
        
        if self.transform:
            image = self.transform(image)
            
        """
            IMPORTANT: Original labels starts from 1. Here we minus 1 !!!
        """
        return torch.tensor(image, dtype=torch.float32), self.labels[idx] - 1
    
# transform = transforms.Compose([
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

def get_imagenet_loaders():
    train_dataset = ImageNet64(data_dir=config.imagenet_root + '/train')
    val_dataset = ImageNet64(data_dir=config.imagenet_root + '/val')
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=6,shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=6,shuffle=False)
    
    return train_loader, val_loader

def get_coco_loaders():
    transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    coco_train_dataset = CocoDetection(root=config.coco_root+'/train', annFile=config.coco_root+'/annotations/train.json', transform=transform)
    coco_val_dataset = CocoDetection(root=config.coco_root+'val', annFile=config.coco_root+'/annotations/val.json', transform=transform)
    
    train_loader = DataLoader(coco_train_dataset, batch_size=2, shuffle=True, collate_fn=torchvision.datasets.CocoDetection.collate_fn)
    val_loader = DataLoader(coco_val_dataset, batch_size=2, shuffle=False, collate_fn=torchvision.datasets.CocoDetection.collate_fn)

    return train_loader, val_loader

def get_dataloaders():
    if config.TASK_TYPE == config.CLS:
        return get_imagenet_loaders()
    else:
        return get_coco_loaders()