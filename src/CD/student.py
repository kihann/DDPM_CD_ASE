import torch
import torch.nn as nn
import math

def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.norm(x).reshape(B, C, H * W)
        q, k, v = self.qkv(x).chunk(3, dim=1)
        attn = (q.transpose(1, 2) @ k) / (C ** .5)
        attn = attn.softmax(dim=-1)
        out = v @ attn.transpose(1, 2)
        return (x + self.proj(out)).reshape(B, C, H, W)

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

class MidBlock(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.block1 = ResidualBlock(channels, channels, emb_dim)
        self.attn = SelfAttention(channels)
        self.block2 = ResidualBlock(channels, channels, emb_dim)

    def forward(self, x, emb):
        x = self.block1(x, emb)
        x = self.attn(x)
        x = self.block2(x, emb)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class UNetStudent(nn.Module):
    def __init__(self, img_channels=3, base_channels=128, emb_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)

        self.down1 = ResidualBlock(base_channels, base_channels * 2, emb_dim)
        self.ds1 = Downsample(base_channels * 2)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 4, emb_dim)
        self.ds2 = Downsample(base_channels * 4)
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 8, emb_dim)
        self.ds3 = Downsample(base_channels * 8)
        self.down4 = ResidualBlock(base_channels * 8, base_channels * 16, emb_dim)
        self.ds4 = Downsample(base_channels * 16)

        self.mid = MidBlock(base_channels * 16, emb_dim)

        self.up4 = Upsample(base_channels * 16)
        self.up_block4 = ResidualBlock(base_channels * 16 + base_channels * 16, base_channels * 8, emb_dim)
        self.exit4 = nn.Conv2d(base_channels * 8, img_channels, 1)
        self.up3 = Upsample(base_channels * 8)
        self.up_block3 = ResidualBlock(base_channels * 8 + base_channels * 8, base_channels * 4, emb_dim)
        self.exit3 = nn.Conv2d(base_channels * 4, img_channels, 1)
        self.up2 = Upsample(base_channels * 4)
        self.up_block2 = ResidualBlock(base_channels * 4 + base_channels * 4, base_channels * 2, emb_dim)
        self.exit2 = nn.Conv2d(base_channels * 2, img_channels, 1)
        self.up1 = Upsample(base_channels * 2)
        self.up_block1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, emb_dim)
        self.exit1 = nn.Conv2d(base_channels, img_channels, 1)

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 1)
        )

    def forward(self, x, t, return_all=False):
        emb = sinusoidal_embedding(t, self.time_mlp[1].in_features)
        emb = self.time_mlp(emb)

        x = self.init_conv(x)
        x1 = self.down1(x, emb)
        x1_d = self.ds1(x1)
        x2 = self.down2(x1_d, emb)
        x2_d = self.ds2(x2)
        x3 = self.down3(x2_d, emb)
        x3_d = self.ds3(x3)
        x4 = self.down4(x3_d, emb)
        x4_d = self.ds4(x4)

        mid = self.mid(x4_d, emb)

        up4 = self.up4(mid)
        up4 = self.up_block4(torch.cat([up4, x4], dim=1), emb)
        up3 = self.up3(up4)
        up3 = self.up_block3(torch.cat([up3, x3], dim=1), emb)
        up2 = self.up2(up3)
        up2 = self.up_block2(torch.cat([up2, x2], dim=1), emb)
        up1 = self.up1(up2)
        up1 = self.up_block1(torch.cat([up1, x1], dim=1), emb)

        pred4 = self.exit4(up4)
        pred3 = self.exit3(up3)
        pred2 = self.exit2(up2)
        pred1 = self.exit1(up1)

        final = self.final_conv(up1)

        if return_all:
            return [pred4, pred3, pred2, pred1, final]
        else:
            return final
