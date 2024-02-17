"""
      This files contains the classes to build a Swin Trnasformer (pdf: https://arxiv.org/abs/2103.14030).
      According to our proposal, we will separately implement Window Partition, Window Reverse, Cyclic Shift as functions, 
    and Patch Merging, Partch Partition, W-MSA, SW-MSA, Swin-Transformer Block and Swin Transformer as nn.Modules.
    
      The default values of the hyperparameters refer to the setting of the original paper.
"""

import torch
import torch.nn as nn
from utils import trunc_normal_

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
    """
          This module downsamples before each stage to reduce the resolution and adjust the number of channels to 
        form a hierarchical design, and also saves some computation.
    """
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
    
def partition_into_windows(image_tensor, window_size):
    """
    Args:
        image_tensor: Input image tensor with shape (B, H, W, C)
        window_size (int): Size of the window for partitioning

    Returns:
        windows: Tensor of windows with shape (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = image_tensor.shape
    
    windows = image_tensor.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows

def reverse_windows_to_image(windows, window_size, image_height, image_width):
    """
    Args:
        windows: Tensor of windows with shape (num_windows * B, window_size, window_size, C)
        window_size (int): Size of the window
        image_height (int): Height of the original image
        image_width (int): Width of the original image

    Returns:
        image_tensor: Reconstructed image tensor with shape (B, image_height, image_width, C)
    """
    B = windows.shape[0] // (image_height // window_size * image_width // window_size)
    num_windows_H = image_height // window_size
    num_windows_W = image_width // window_size

    image_tensor = windows.reshape(B, num_windows_H, num_windows_W, window_size, window_size, -1)
    image_tensor = image_tensor.permute(0, 1, 3, 2, 4, 5).reshape(B, image_height, image_width, -1)
    return image_tensor

class WindowAttention(nn.Module):
    """
          Compute self-attention within local windows, which is similar to ViT. The windows are arranged to 
        evenly partition the image in a non-overlapping manner.
          This module supports both `Window-based` and `Shifted-Window-based` attention, because window partition,
        including the `shifted` style, is finished before calling this module.
    """
    def __init__(self, embedding_dim, window_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (embedding_dim // num_heads) ** (-0.5)
        
        # Linear transformations
        self.qkv = nn.Linear(self.embedding_dim, self.embedding_dim * 3)
        self.linear = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.linear_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim = -1)
        self.init_relative_pos()
       
    def init_relative_pos(self):
        """
            Compute pair-wise relative position index for each token inside the window.
            This function refers to the official code.
        """
        # Define a parameter table of relative position bias
        num_relative_positions = (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(num_relative_positions, self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # Compute relative position indices
        coords_h, coords_w = torch.meshgrid(torch.arange(self.window_size[0]), torch.arange(self.window_size[1]))
        relative_coords = (coords_h.flatten() - coords_h.flatten().unsqueeze(1)).unsqueeze(-1)
        relative_coords += self.window_size[0] - 1
        relative_coords *= 2 * self.window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))  # Wh*Ww, Wh*Ww

        # Initialize relative position bias
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        nw_B, N, C = x.shape
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)    # self.qkv(x) returns shape (embed * 3)

        q *= self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], 
                                                            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)

        attn += relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn.view(nw_B // mask.shape[0], mask.shape[0], self.num_heads, N, N) + \
                                mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(nw_B, N, C)
        x = self.linear_drop(self.linear(x))
        return x