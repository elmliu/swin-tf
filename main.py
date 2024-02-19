import torch
import torch.nn as nn
from torch.optim import Adam
from datahelpers import get_imagenet_loaders
import config
from tqdm import tqdm
from models import SwinTransformer

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
            outputs = model(images)
            pred = torch.max(outputs, 1).cpu().flatten()
            labels = labels.cpu().flatten()
            
            total += len(labels)
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
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print('Average loss:', total_loss/len(train_loader))
        
        if epoch % config.test_ep_gap == 0 or epoch == config.epochs:
            acc = test(model, test_loader)
            print(f"Epoch {epoch} | Acc {acc:.4f}")
            
def get_model():
    model = SwinTransformer(img_size=64, stage_blocks=config.SF_SIZE['stage_blocks'], 
                            window_size=4)  # Set window size to 4, not default value 7
    return model

if __name__ == '__main__':
    model = get_model()
    train_loader, test_loader = get_imagenet_loaders()
    train(model, train_loader, test_loader)