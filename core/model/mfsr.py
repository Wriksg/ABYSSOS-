import torch
import torch.nn as nn
from .sar_gate import SARGate

class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class MFSR(nn.Module):
    def __init__(self, num_optical_bands=4, num_sar_bands=2, scale=4):
        super().__init__()
        self.scale = scale
        
        # 1. Shared Encoder: frame_i (C), ref_frame (C), mask_i (1) -> (2C + 1)
        in_channels = (num_optical_bands * 2) + 1
        self.encoder_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_res = nn.Sequential(*[ResBlock(64) for _ in range(4)])

        # 2. Recursive Fusion Blocks
        self.fuse_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.fuse_res = ResBlock(64)

        # 3. SAR Encoder & Gate
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(num_sar_bands, 64, kernel_size=3, padding=1),
            ResBlock(64)
        )
        self.sar_gate = SARGate()

        # 4. Decoder to 2.5m (4x upsample)
        self.decoder_res = nn.Sequential(*[ResBlock(64) for _ in range(4)])
        # C * scale^2 channels needed for PixelShuffle
        self.decoder_conv = nn.Conv2d(64, num_optical_bands * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def fuse_block(self, x):
        """Processes a concatenated pair of 64-channel features (128 -> 64)"""
        x = self.fuse_conv(x)
        x = self.fuse_res(x)
        return x

    def fuse(self, feats):
        """Recursively fuses a list of feature maps of any length K"""
        while len(feats) > 1:
            nxt = []
            for i in range(0, len(feats) - 1, 2):
                nxt.append(self.fuse_block(torch.cat([feats[i], feats[i+1]], 1)))
            if len(feats) % 2 != 0:
                nxt.append(feats[-1])
            feats = nxt
        return feats[0]

    def forward(self, lrs, masks, sar, clear_fraction):
        """
        lrs: (B, K, C, H, W) -> Multi-frame optical stack
        masks: (B, K, 1, H, W) -> 1=clear, 0=cloud
        sar: (B, C_sar, H, W) -> Single SAR composite
        clear_fraction: (B,) -> Used for gating SAR
        """
        B, K, C, H, W = lrs.shape
        
        # We always treat the first frame (i=0) as the reference frame
        ref_frame = lrs[:, 0]
        
        # 1. Encode all K frames
        encoded_feats = []
        for i in range(K):
            frame_i = lrs[:, i]
            mask_i = masks[:, i]
            
            # Concatenate frame_i, ref_frame, and mask_i
            x = torch.cat([frame_i, ref_frame, mask_i], dim=1)
            x = self.encoder_conv(x)
            x = self.encoder_res(x)
            encoded_feats.append(x)

        # 2. Recursively fuse the optical features
        fused_optical = self.fuse(encoded_feats)

        # 3. Process SAR and Gate
        sar_feat = self.sar_encoder(sar)
        fused_feat, gate_val = self.sar_gate(fused_optical, sar_feat, clear_fraction)

        # 4. Decode and Upsample
        out = self.decoder_res(fused_feat)
        out = self.decoder_conv(out)
        sr = self.pixel_shuffle(out)

        return sr, gate_val