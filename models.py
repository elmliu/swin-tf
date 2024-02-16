"""
      This files contains the classes to build a Swin Trnasformer (pdf: https://arxiv.org/abs/2103.14030).
      According to our proposal, we will separately implement Patch Merging and Cyclic Shift as functions, 
    and Partch Partition, W-MSA, SW-MSA, Swin-Transformer Block and Swin Transformer as nn.Modules.
    
      The default values of the hyperparameters refer to the setting of the original paper.
"""

import torch
import torch.nn as nn

class PatchPartition(nn.Module):
    """
        Embed the original image by cropping it into patches of size patch_size * patch_size.
        Implemented using 2D convolutional layers and dimension unfolding.
    """

    def __init__(self, image_size=224, patch_size=4, in_channels=3, embedding_dim=96, norm_layer=None):
        super().__init__()

        self.image_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.conv_layer = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.conv_layer(x)              # Output shape: (N, embedding_dim, H_out, W_out)
        x = torch.flatten(x, 2)       # Flatten H, W dimensions: (N, embedding_dim, num_patches)
        x = torch.transpose(x, 1, 2)  # Transpose to have channels last: (N, num_patches, embedding_dim)
        if self.norm is not None:
            x = self.norm(x)
        return x
