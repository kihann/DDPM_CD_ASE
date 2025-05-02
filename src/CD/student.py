import torch
import torch.nn as nn
import math

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.activation = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.activation(self.norm2(h)))
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.op = nn.Conv2d(in_channels, in_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class UNetStudent(nn.Module):
    def __init__(self, img_channels=3, base_channels=64, emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels * 2, emb_dim)
        self.ds1 = Downsample(base_channels * 2)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, emb_dim)
        self.ds2 = Downsample(base_channels * 4)

        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, emb_dim)

        self.up2 = Upsample(base_channels * 4)
        self.up_block2 = ResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2, emb_dim)
        self.up1 = Upsample(base_channels * 2)
        self.up_block1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, emb_dim)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 1)
        )

    def forward(self, x, t):
        emb = sinusoidal_embedding(t, self.time_mlp[0].in_features)
        emb = self.time_mlp(emb)

        x1 = self.init_conv(x)
        x2 = self.down1(x1, emb)
        x2_d = self.ds1(x2)
        x3 = self.down2(x2_d, emb)
        x3_d = self.ds2(x3)

        mid = self.mid(x3_d, emb)

        up2 = self.up2(mid)
        up2 = self.up_block2(torch.cat([up2, x3], dim=1), emb)
        up1 = self.up1(up2)
        up1 = self.up_block1(torch.cat([up1, x2], dim=1), emb)

        return self.final_conv(up1)
