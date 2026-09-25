import torch
import torch.nn as nn

class SARGate(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialized so that if clear_fraction is 1.0, g is low (~0.26)
        # and if clear_fraction is 0.0, g is high (~0.73)
        self.w = nn.Parameter(torch.tensor(2.0))
        self.b = nn.Parameter(torch.tensor(-1.0))
        
    def forward(self, optical_feat, sar_feat, clear_fraction):
        """
        optical_feat: (B, 64, H, W)
        sar_feat: (B, 64, H, W)
        clear_fraction: (B,) float between 0 (fully cloudy) and 1 (fully clear)
        """
        # Calculate gate value for each batch item
        g = torch.sigmoid(self.w * (1 - clear_fraction) + self.b)
        
        # Reshape g to broadcast across spatial and channel dimensions
        # Output is optical + gated SAR
        return optical_feat + g.view(-1, 1, 1, 1) * sar_feat, g