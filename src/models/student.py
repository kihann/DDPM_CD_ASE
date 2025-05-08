import torch
import torch.nn as nn
import math
import utils

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
    def __init__(self, img_channels=3, base_channels=128, emb_dim=256, 
                 diffusion_steps=utils.TIMESTEP, skip_exponent=2.5, skip_thresholds=None):
        super().__init__()
        self.total_steps = diffusion_steps
        self.skip_exponent = skip_exponent
        self.skip_thresholds = skip_thresholds or {
            4: 0.3,
            3: 0.5,
            2: 0.7,
            1: 0.9,
        }
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

        # Learnable upsampling blocks for each exit to align outputs with final 256x256 resolution
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(img_channels, img_channels, 4, stride=2, padding=1),
            nn.ConvTranspose2d(img_channels, img_channels, 4, stride=2, padding=1),
            nn.ConvTranspose2d(img_channels, img_channels, 4, stride=2, padding=1),
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(img_channels, img_channels, 4, stride=2, padding=1),
            nn.ConvTranspose2d(img_channels, img_channels, 4, stride=2, padding=1),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(img_channels, img_channels, 4, stride=2, padding=1),
        )

        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, img_channels, 1)
        )

        # Projection layers for skipped blocks
        self.skip_proj4 = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=1)
        self.skip_proj3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        self.skip_proj2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.skip_proj1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)

    def should_skip_block(self, t, level):
        if isinstance(t, torch.Tensor):
            t_val = t[0].item() if t.ndim > 0 else t.item()
        else:
            t_val = t

        p = (t_val / self.total_steps) ** self.skip_exponent
        return p > self.skip_thresholds.get(level, 1.0)

    def forward(self, x, t, return_all=False):
        emb = sinusoidal_embedding(t, self.time_mlp[1].in_features)
        emb = self.time_mlp(emb)

        # Downsampling path (기존과 동일)
        x_init = self.init_conv(x) # base_channels (128)
        x1 = self.down1(x_init, emb)    # base_channels * 2 (256)
        x1_d = self.ds1(x1)
        x2 = self.down2(x1_d, emb)    # base_channels * 4 (512)
        x2_d = self.ds2(x2)
        x3 = self.down3(x2_d, emb)    # base_channels * 8 (1024)
        x3_d = self.ds3(x3)
        x4 = self.down4(x3_d, emb)    # base_channels * 16 (2048)
        x4_d = self.ds4(x4)

        mid_out = self.mid(x4_d, emb) # base_channels * 16 (2048)

        # Upsampling path with skip projections
        # Level 4
        u_l4_transposed = self.up4(mid_out) # In/Out: base_channels * 16 (2048)
        if not self.should_skip_block(t, 4):
            h_l4 = self.up_block4(torch.cat([u_l4_transposed, x4], dim=1), emb) # Out: base_channels * 8 (1024)
        else:
            h_l4 = self.skip_proj4(u_l4_transposed) # Out: base_channels * 8 (1024)

        pred4 = self.exit4(h_l4)
        upsampled4 = self.upsample4(pred4)

        # Level 3
        u_l3_transposed = self.up3(h_l4) # In/Out: base_channels * 8 (1024)
        if not self.should_skip_block(t, 3):
            h_l3 = self.up_block3(torch.cat([u_l3_transposed, x3], dim=1), emb) # Out: base_channels * 4 (512)
        else:
            h_l3 = self.skip_proj3(u_l3_transposed) # Out: base_channels * 4 (512)

        pred3 = self.exit3(h_l3)
        upsampled3 = self.upsample3(pred3)

        # Level 2
        u_l2_transposed = self.up2(h_l3) # In/Out: base_channels * 4 (512)
        if not self.should_skip_block(t, 2):
            h_l2 = self.up_block2(torch.cat([u_l2_transposed, x2], dim=1), emb) # Out: base_channels * 2 (256)
        else:
            h_l2 = self.skip_proj2(u_l2_transposed) # Out: base_channels * 2 (256)

        pred2 = self.exit2(h_l2)
        upsampled2 = self.upsample2(pred2)

        # Level 1
        u_l1_transposed = self.up1(h_l2) # In/Out: base_channels * 2 (256)
        if not self.should_skip_block(t, 1):
            h_l1 = self.up_block1(torch.cat([u_l1_transposed, x1], dim=1), emb) # Out: base_channels (128)
        else:
            h_l1 = self.skip_proj1(u_l1_transposed) # Out: base_channels (128)

        pred1 = self.exit1(h_l1)
        # upsampled1 is not typically defined this way, pred1 is usually at a feature size for final_conv

        final_out = self.final_conv(h_l1)

        if return_all:
            return [upsampled4, upsampled3, upsampled2, pred1, final_out]
        else:
            return final_out
