import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import warnings
import torch

# Edit this path to your work dir
import sys
sys.path.append("/data/ljc/swin-tf")
from main import get_model

warnings.filterwarnings("ignore", category=UserWarning)


def create_model(num_classes,checkpoint=None,device='cpu'):
    """
    Create a model for object detection using the Faster R-CNN architecture.

    Parameters:
    - num_classes (int): The number of classes for object detection. (Total classes + 1 [for background class])
    - checkpoint (str) : checkpoint path for the pretrained custom model
    - device (str) : cpu / cuda
    Returns:
    - model (torchvision.models.detection.fasterrcnn_resnet50_fpn): The created model for object detection.
    """
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    #     pretrained=True,
    #     weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    #     pretrained_backbone=True,
    #     #weights_backbone = 'ResNet50_Weights.DEFAULT',
    # )
    
    """
        Use my SwinTransformer and FasterRCNN to create a model for object detection
    """
    swin_backbone = get_model('swin')
    backbone = resnet_fpn_backbone('resnet50')
    model = FasterRCNN(backbone, num_classes=91)

    # 替换 Faster R-CNN 的 backbone 为 Swin Transformer
    model.backbone = swin_backbone
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = model.to(device)
    return model
