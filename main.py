import torch
import torch.nn as nn
from torch.optim import Adam
from datahelpers import get_imagenet_loaders
import config
import numpy as np
from tqdm import tqdm
from models import SwinTransformer

from torchvision.models import swin_t
from transformers import SwinForImageClassification, SwinConfig

DEVICE = 'cuda'

"""
    Calculate top-1 accuracy.
"""
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(DEVICE)
            outputs = model(images)
            
            if isinstance(model, SwinForImageClassification):
                pred = outputs.logtis.cpu().argmax(-1).flatten()
            else:
                _, pred = torch.max(outputs.cpu(), 1)
            # labels = labels.flatten()
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    top1_accuracy = 100 * correct / total
    return top1_accuracy

def train(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.lr)
    model.train()
    
    for ep in range(config.epochs):
        epoch = ep + 1
        print('Epoch', epoch)
        
        total_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # print(images.shape)
            # print(labels.shape)
            # print(images)
            # print(labels)
            
            if isinstance(model, SwinForImageClassification):
                outputs = model(pixel_values=images, labels = labels, return_dict=True)
                # print(outputs)
                loss = outputs.loss
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # print(loss.item())            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print('Average loss:', total_loss/len(train_loader))
        
        if epoch % config.test_ep_gap == 0 or epoch == config.epochs:
            acc = test(model, test_loader)
            print(f"Epoch {epoch} | Acc {acc:.4f}")
            
def get_model():
    # model = SwinTransformer(img_size=64, stage_blocks=config.SF_SIZE['stage_blocks'], 
    #                         window_size=8, 
    #                         patch_size=4, 
    #                         embedding_dim=config.SF_SIZE['embed_dim'])
                            
    # model = SwinTransformer(stage_blocks=config.SF_SIZE['stage_blocks'])  # For debug only
    
    # official model
    # model = swin_t()
    
    conf = SwinConfig(image_size = 64, patch_size = 2, window_size = 4, num_labels=1000)
    model = SwinForImageClassification(conf)
    
    return model

if __name__ == '__main__':
    model = get_model().to(DEVICE)
    train_loader, test_loader = get_imagenet_loaders()
    train(model, train_loader, test_loader)