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

    def __init__(self, img_size=224, patch_size=4, in_channels=3, embedding_dim=96):
        super().__init__()

        self.image_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patch_res = [self.image_size[0] // self.patch_size[0], 
                          self.image_size[1] // self.patch_size[1]]

        self.in_channels = in_channels
        self.embedding_dim = embedding_dim

        self.conv_layer = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # print(x.dtype)
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
    
    # windows = image_tensor.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    # windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    
    # Use official code
    image_tensor = image_tensor.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = image_tensor.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    
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
        # q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)    # self.qkv(x) returns shape (embed * 3)

        # Below is the official code
        qkv = self.qkv(x).reshape(nw_B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size[0] * self.window_size[1], 
                                                            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)

        attn = attn + relative_position_bias.unsqueeze(0)

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
    
class SwinTransBlock(nn.Module):
    """
        Block sturcture: LayerNorm1 -> WindowAttention -> ResidualConnecttion -> 
                            LayerNorm2 -> MLP -> ResidualConnection
    """
    def __init__(self, dim, input_res, num_heads, window_size=7, shift_size=0,
                 mlp_hid_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        
        self.dim = dim
        self.input_res = input_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        # Below is from the official code
        if min(self.input_res) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_res)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), 
            num_heads=num_heads,
            attn_drop=attn_drop, 
            proj_drop=drop)

        mlp_hidden_dim = int(dim * mlp_hid_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

        self.attn_mask = self.get_shift_mask()
        
        # Move to cuda
        if self.attn_mask is not None:  
            self.attn_mask = self.attn_mask.to('cuda')

    # This part refers to the official code.
    def get_shift_mask(self):
        if self.shift_size > 0:
            # calculate cyclic shift attention mask for SW-MSA
            H, W = self.input_res
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = partition_into_windows(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            # Do nothing.
            attn_mask = None
            
        return attn_mask
    
    def get_shifted_windows(self, x):
        if self.shift_size > 0:
            # cyclic shift
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # partition windows
        return partition_into_windows(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

    def rev_shifted_windows(self, attn_windows, H, W):
        shifted_x = reverse_windows_to_image(attn_windows, self.window_size, H, W)  # B H' W' C

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        return x
        

    def forward(self, x):
        H, W = self.input_res
        B, L, C = x.shape
        
        # print('input_res:', self.input_res)
        # print('x_shape:', x.shape)
        
        # From official code        
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        """
            Go through LayerNorm 1.
        """
        x = self.norm1(x).view(B, H, W, C)      
        
        """
            Go througn Window based Attention (shift if necessary).
        """
        # Shift the windows if self.shift_size > 0, using the cyclic shift
        x_windows = self.get_shifted_windows(x) \
                        .view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Calculate the Windows Attention
        attn_windows = self.attn(x_windows, mask=self.attn_mask) \
                        .view(-1, self.window_size, self.window_size, C)  

        # reverse cyclic shift
        x = self.rev_shifted_windows(attn_windows, H, W)\
                        .view(B, H * W, C)
        
        """
            Residual connection 1
        """
        x = shortcut + x

        """
            Go through LayerNorm 1, MLP, and residual connection 2
        """
        x = x + self.mlp(self.norm2(x))

        return x
    
class SwinStageLayer(nn.Module):
    """
        A complete Swin Transformer is consisted of four stages, and one SwinStageLayer is consisted of 2*k Swin Transformer blocks.
        Swin Transformer blocks always appear as a combination of repeated non-shifted and shifted pattern.
    """
    
    def __init__(self, dim, input_res, num_blocks, num_heads, window_size, is_last_stage,
                 mlp_hid_ratio=4., drop=0., attn_drop=0.):

        super().__init__()
        
        assert num_blocks % 2 == 0, "num_blocks for SwinStageLayer has to be an even number."
        
        self.dim = dim
        self.input_res = input_res
        self.num_blocks = num_blocks

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransBlock(dim=dim, input_res=input_res,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size = 0 if (i % 2 == 0) else window_size // 2,  # Repeated non-shifted and shifted pattern.
                                 mlp_hid_ratio=mlp_hid_ratio,
                                 drop=drop, attn_drop=attn_drop)
            for i in range(num_blocks)])
        
        # Create PatchMerging if this is not the last stage layer
        if not is_last_stage:
            self.patch_merging = PatchMerging(patch_shape=input_res, in_channels=dim)
        else:
            self.patch_merging = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
            
        if self.patch_merging is not None:
            x = self.patch_merging(x)
        return x
    
class SwinTransformer(nn.Module):
    """
        SwinTransformer architecture:
            PatchPartition -> Four Stage Layers (PatchMerging + 2 * SwinTransBlocks) -> Linear Classifier
    """
    def __init__(self, img_size=224, patch_size=4, in_channels=3, n_classes=1000,
                 embedding_dim=96, stage_blocks=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_hid_ratio=4., attn_drop_rate=0.):
        """
            For ImageNet-1k, n_classes = 1000
        """
        super().__init__()

        self.n_classes = n_classes
        self.n_stages = len(stage_blocks)
        self.embedding_dim = embedding_dim
        self.n_feat = int(embedding_dim * 2 ** (self.n_stages - 1))
        self.mlp_hid_ratio = mlp_hid_ratio

        # split image into non-overlapping patches
        self.patch_partition = PatchPartition(
                                img_size=img_size, 
                                patch_size=patch_size, 
                                in_channels=in_channels, 
                                embedding_dim=embedding_dim)

        # build layers
        self.stage_layers = nn.ModuleList()
        patch_res = self.patch_partition.patch_res
        
        for lay_idx in range(self.n_stages):
            input_res = (patch_res[0] // (2 ** lay_idx),
                        patch_res[1] // (2 ** lay_idx))
            
            stage = SwinStageLayer(dim=int(embedding_dim * 2 ** lay_idx),
                               input_res=input_res,
                               num_blocks=stage_blocks[lay_idx],
                               num_heads=num_heads[lay_idx],
                               window_size=window_size,
                               is_last_stage=(lay_idx == self.n_stages-1),
                               mlp_hid_ratio=self.mlp_hid_ratio,
                               attn_drop=attn_drop_rate)
            self.stage_layers.append(stage)

        self.layer_norm = nn.LayerNorm(self.n_feat)
        self.global_avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(self.n_feat, n_classes) if n_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        
    # This function refers to the official code
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def extract_features(self, x):
        # print(x.dtype)
        x = self.patch_partition(x)

        for stage in self.stage_layers:
            x = stage(x)

        x = self.layer_norm(x)  # B L C
        x = self.global_avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
        return x