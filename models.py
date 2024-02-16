"""
      This files contains the classes to build a Swin Trnasformer (pdf: https://arxiv.org/abs/2103.14030).
      According to our proposal, we will separately implement Cyclic Shift as functions, 
    and Patch Merging, Partch Partition, W-MSA, SW-MSA, Swin-Transformer Block and Swin Transformer as nn.Modules.
    
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
    
class PatchMerging(nn.Module):
    def __init__(self, patch_shape, in_channels):
        """
        Args:
            patch_shape (tuple(int, int)): Shape of the input image patches (H, W).
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.patch_shape = patch_shape
        self.in_channels = in_channels

        # Reduction layer reduces the dimension of the input features
        self.linear_reduc = nn.Linear(4 * in_channels, 2 * in_channels, bias=False)

        # Normalization layer
        self.norm = nn.LayerNorm(4 * in_channels)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (N, H*W, C), where N is batch size,
                              H * W is the number of patches, and C is the number of input channels.
        Returns:
            torch.Tensor: Output tensor after patch merging.
        """
        H, W = self.patch_shape
        B, C = x.shape[0], x.shape[2]

        # Reshape input tensor
        x = x.view(B, H, W, C)

        # Patch merging. This part refers to the official code.
        # Selects all batches, even-indexed rows in height, and even-indexed columns in width.
        x0 = x[:, 0::2, 0::2, :]
        # Selects all batches, odd-indexed rows in height, and even-indexed columns in width.
        x1 = x[:, 1::2, 0::2, :]
        # Selects all batches, even-indexed rows in height, and odd-indexed columns in width.
        x2 = x[:, 0::2, 1::2, :]
        # Selects all batches, odd-indexed rows in height, and odd-indexed columns in width.
        x3 = x[:, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)  # Shape: (B, H/2*W/2, 4*C)

        # Normalization and dimension reduction
        return self.linear_reduc(self.norm(x))