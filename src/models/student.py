import torch
import torch.nn as nn
import math

STUDENT_SIGMA_MIN_DEFAULT = 0.002
STUDENT_SIGMA_MAX_DEFAULT = 80.0

def sinusoidal_embedding_continuous(sigmas: torch.Tensor, dim: int, max_period=10000.0):
    """Generates Sinusoidal Positional Embedding for continuous sigma values."""
    if sigmas.ndim == 0:
        sigmas = sigmas.unsqueeze(0)
    if sigmas.ndim == 1:
        sigmas = sigmas.unsqueeze(-1)
        
    log_sigmas = torch.log(sigmas.float())

    half_dim = dim // 2

    # Case 1: Embedding dimension is too small
    if half_dim == 0: # dim is 0 or 1
        return torch.zeros(*sigmas.shape[:-1], dim, device=sigmas.device, dtype=log_sigmas.dtype)

    # Case 2: half_dim is 1 (original dim is 2 or 3)
    # The standard formula for emb_freqs would lead to division by zero or problematic log.
    # For such low dimensions, a simpler or alternative embedding might be needed.
    # Here, we'll create a constant or simple frequency.
    # Or, as a simple fix for now, ensure dim is large enough before calling this.
    # The original code had torch.zeros, let's stick to a more robust calculation or clear handling.
    if half_dim == 1: # dim is 2 or 3
        # Fallback: create a single frequency, e.g., based on a fixed small value or 1.0
        # This part might need more thought if dim=2 or 3 is common and important.
        # For now, let's make it a single frequency component.
        # Or, if only half_dim-1 = 0 is the issue for log, then handle that.
        # The issue is math.log(max_period) / (half_dim - 1)
        # If half_dim = 1, then half_dim - 1 = 0.
        # A simple fallback for half_dim = 1 might be to use a single, predefined frequency.
        # For example, just `torch.ones(1, device=sigmas.device)` or a small constant.
        # Let's assume dim will be >= 4 for this embedding to work as intended with multiple frequencies.
        # If dim=2, half_dim=1. torch.arange(1) is [0]. exp(0 * -val) is exp(0) = 1.
        # So emb_freqs would be tensor([1.])
        # This will make emb_args = log_sigmas * 1.
        # This is a valid, though simple, embedding for dim=2.
        emb_freqs = torch.ones(half_dim, device=sigmas.device) # Results in [cos(log_s), sin(log_s)] if dim=2
    
    # Case 3: Normal calculation for half_dim > 1 (dim >= 4)
    else: # half_dim > 1
        emb_freqs_val = math.log(max_period) / (half_dim - 1)
        emb_freqs = torch.exp(torch.arange(half_dim, device=sigmas.device) * -emb_freqs_val)

    # Common part for emb_freqs processing and final embedding generation
    emb_freqs = emb_freqs.unsqueeze(0)  # Shape: [1, half_dim]
    emb_args = log_sigmas * emb_freqs  # Shape: [B, half_dim]
    
    embedding = torch.cat([torch.sin(emb_args), torch.cos(emb_args)], dim=-1)  # Shape: [B, dim or dim-1]
    
    if dim % 2 == 1: # Pad if original dim was odd
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
    return embedding

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

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_reshaped = self.norm(x).view(B, C, H * W)
        q, k, v = self.qkv(x_reshaped).chunk(3, dim=1)
        attn_weights = torch.bmm(q.transpose(1, 2), k) * (C ** -0.5)
        attn_weights = attn_weights.softmax(dim=-1)
        out_reshaped = torch.bmm(v, attn_weights.transpose(1, 2))
        return (x_reshaped + self.proj(out_reshaped)).view(B, C, H, W)


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
    def __init__(self, img_channels=3, base_channels=64, emb_dim=256,
                 sigma_min=STUDENT_SIGMA_MIN_DEFAULT, sigma_max=STUDENT_SIGMA_MAX_DEFAULT,
                 skip_exponent=2.5, skip_thresholds=None):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)
        self.skip_exponent = skip_exponent
        self.skip_thresholds = skip_thresholds or {
            4: 0.3, 3: 0.5, 2: 0.7, 1: 0.9,
        }

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4), # Standard practice: expand then contract
            nn.SiLU(),
            nn.Linear(emb_dim * 4, emb_dim),
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
        
        self.skip_proj4 = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=1)
        self.skip_proj3 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        self.skip_proj2 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        self.skip_proj1 = nn.Conv2d(base_channels * 2, base_channels, kernel_size=1)


    def should_skip_block(self, sigmas_batch: torch.Tensor, level: int) -> torch.Tensor:
        """Determines if blocks should be skipped for a batch of sigmas at a given level."""
        log_sigmas = torch.log(sigmas_batch.float())
        
        if self.log_sigma_max == self.log_sigma_min:
            normalized_log_sigmas = torch.full_like(sigmas_batch, 0.5)
        else:
            normalized_log_sigmas = (log_sigmas - self.log_sigma_min) / (self.log_sigma_max - self.log_sigma_min)
        
        normalized_log_sigmas = torch.clamp(normalized_log_sigmas, 0.0, 1.0)
        
        p_values = normalized_log_sigmas ** self.skip_exponent
        
        threshold = self.skip_thresholds.get(level, 1.0)
        return p_values > threshold

    def forward(self, x: torch.Tensor, sigmas: torch.Tensor, return_all=False):
        # sigmas: [B] tensor of continuous sigma values
        # emb_dim is the output dimension of sinusoidal_embedding_continuous
        # and input dimension to self.time_mlp's first Linear layer.
        # Assuming self.time_mlp[0] is Linear layer if no LayerNorm, or self.time_mlp[1] if LayerNorm is first.
        # Based on current MLP: nn.Linear(emb_dim, emb_dim * 4) -> emb_dim should be correct.
        current_emb_dim_for_sinusoidal = self.time_mlp[0].in_features # emb_dim * 4 if first layer is expand
                                                                    # should be emb_dim if first layer takes emb_dim
        # Let's assume sinusoidal_embedding_continuous output dim matches the UNet's emb_dim directly
        emb = sinusoidal_embedding_continuous(sigmas, self.down1.emb_proj.in_features)
        emb = self.time_mlp(emb)

        # Downsampling path
        x_init = self.init_conv(x)
        x1 = self.down1(x_init, emb); # print(f"x1: {x1.shape}")
        x1_d = self.ds1(x1); # print(f"x1_d: {x1_d.shape}")
        x2 = self.down2(x1_d, emb); # print(f"x2: {x2.shape}")
        x2_d = self.ds2(x2); # print(f"x2_d: {x2_d.shape}")
        x3 = self.down3(x2_d, emb); # print(f"x3: {x3.shape}")
        x3_d = self.ds3(x3); # print(f"x3_d: {x3_d.shape}")
        x4 = self.down4(x3_d, emb); # print(f"x4: {x4.shape}")
        x4_d = self.ds4(x4); # print(f"x4_d: {x4_d.shape}")

        mid_out = self.mid(x4_d, emb); # print(f"mid_out: {mid_out.shape}")

        # Upsampling path with adaptive exiting
        # skip_level_X will be a boolean tensor of shape [B]
        skip_level4 = self.should_skip_block(sigmas, 4).view(-1, 1, 1, 1) # Reshape for broadcasting
        u_l4_transposed = self.up4(mid_out)
        h_l4_skip = self.skip_proj4(u_l4_transposed)
        h_l4_full = self.up_block4(torch.cat([u_l4_transposed, x4], dim=1), emb)
        h_l4 = torch.where(skip_level4, h_l4_skip, h_l4_full)
        pred4 = self.exit4(h_l4); # print(f"pred4: {pred4.shape}")
        upsampled4 = self.upsample4(pred4)

        skip_level3 = self.should_skip_block(sigmas, 3).view(-1, 1, 1, 1)
        u_l3_transposed = self.up3(h_l4)
        h_l3_skip = self.skip_proj3(u_l3_transposed)
        h_l3_full = self.up_block3(torch.cat([u_l3_transposed, x3], dim=1), emb)
        h_l3 = torch.where(skip_level3, h_l3_skip, h_l3_full)
        pred3 = self.exit3(h_l3); # print(f"pred3: {pred3.shape}")
        upsampled3 = self.upsample3(pred3)

        skip_level2 = self.should_skip_block(sigmas, 2).view(-1, 1, 1, 1)
        u_l2_transposed = self.up2(h_l3)
        h_l2_skip = self.skip_proj2(u_l2_transposed)
        h_l2_full = self.up_block2(torch.cat([u_l2_transposed, x2], dim=1), emb)
        h_l2 = torch.where(skip_level2, h_l2_skip, h_l2_full)
        pred2 = self.exit2(h_l2); # print(f"pred2: {pred2.shape}")
        upsampled2 = self.upsample2(pred2)

        skip_level1 = self.should_skip_block(sigmas, 1).view(-1, 1, 1, 1)
        u_l1_transposed = self.up1(h_l2)
        h_l1_skip = self.skip_proj1(u_l1_transposed)
        h_l1_full = self.up_block1(torch.cat([u_l1_transposed, x1], dim=1), emb)
        h_l1 = torch.where(skip_level1, h_l1_skip, h_l1_full)
        pred1 = self.exit1(h_l1); # print(f"pred1: {pred1.shape}")
        # pred1 is at the same resolution as final_out, no further upsampling like others

        final_out = self.final_conv(h_l1); # print(f"final_out: {final_out.shape}")

        if return_all:
            # Ensure all returned predictions are for epsilon (or x0, consistently)
            return [upsampled4, upsampled3, upsampled2, pred1, final_out]
        else:
            return final_out