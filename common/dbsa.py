"""Proposed Dual-Branch Spatial Attention (DBSA) block and HTA-ConvNet (Methods, Eqs. 1-9).

Variants used in the manuscript:
  attention  : 'dense' (proposed), 'window' (W-MSA 7x7), 'sra4', 'sra8' (PVT spatial reduction), 'linear'
  fusion     : 'gate' (proposed), 'avg' (G=0.5), 'add', 'concat' (concat + 1x1 conv), 'gate_nores'
  branches   : 'both' (DBSA), 'conv' (DWSC only), 'attn' (MHSA only)
  module     : 'dbsa' (proposed), 'se', 'cbam', 'mobilevit' (replacement ablation)
"""
import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ============================================================================ attention branches
class MHSA(nn.Module):
    """Multi-head self-attention over the H*W tokens (Eqs. 5-7). d_s = k*ceil(C/k), d_k = d_s/k.

    attention='dense'  : full softmax attention (n^2 cost)
    attention='window' : non-overlapping MxM windows (Swin W-MSA, single layer, no shift)
    attention='sraR'   : keys/values from an RxR strided conv-reduced map (PVT SRA)
    attention='linear' : kernelised linear attention, phi(x)=elu(x)+1 (Katharopoulos et al.)
    """

    def __init__(self, channels: int, heads: int = 8, attention: str = "dense", window: int = 7,
                 use_sdpa: bool = False, attn_dropout: float = 0.0):
        super().__init__()
        self.c, self.k = channels, heads
        self.d = heads * math.ceil(channels / heads)
        self.dk = self.d // heads
        self.attention, self.window, self.use_sdpa = attention, window, use_sdpa
        self.norm = nn.LayerNorm(channels)
        self.q = nn.Linear(channels, self.d)
        self.kv = nn.Linear(channels, 2 * self.d)
        self.proj = nn.Linear(self.d, channels)
        self.drop = nn.Dropout(attn_dropout)
        self.sr = None
        if attention.startswith("sra"):
            r = int(attention[3:])
            self.r = r
            self.sr = nn.Conv2d(channels, channels, kernel_size=r, stride=r)
            self.sr_norm = nn.LayerNorm(channels)
        self.last_attn = None  # stored only when requested (analysis)

    def _heads(self, t, n):  # (B, n, d) -> (B, k, n, dk)
        return t.view(t.shape[0], n, self.k, self.dk).transpose(1, 2)

    def _softmax_attn(self, q, k, v):
        if self.use_sdpa:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        a = (q @ k.transpose(-2, -1)) * (self.dk ** -0.5)
        a = self.drop(a.softmax(dim=-1))
        return a @ v

    def forward(self, x):  # x: (B, C, H, W) -> (B, C, H, W)
        B, C, H, W = x.shape
        tok = self.norm(x.flatten(2).transpose(1, 2))  # (B, n, C), LN(Flatten(X))
        n = H * W
        if self.attention == "dense":
            q = self._heads(self.q(tok), n)
            k, v = self.kv(tok).chunk(2, dim=-1)
            out = self._softmax_attn(q, self._heads(k, n), self._heads(v, n))
            out = out.transpose(1, 2).reshape(B, n, self.d)
        elif self.attention.startswith("sra"):
            q = self._heads(self.q(tok), n)
            xr = self.sr(tok.transpose(1, 2).reshape(B, C, H, W))
            m = xr.shape[2] * xr.shape[3]
            xr = self.sr_norm(xr.flatten(2).transpose(1, 2))
            k, v = self.kv(xr).chunk(2, dim=-1)
            out = self._softmax_attn(q, self._heads(k, m), self._heads(v, m))
            out = out.transpose(1, 2).reshape(B, n, self.d)
        elif self.attention == "window":
            M = self.window
            t = tok.transpose(1, 2).reshape(B, C, H, W)
            ph, pw = (M - H % M) % M, (M - W % M) % M
            t = F.pad(t, (0, pw, 0, ph))
            Hp, Wp = H + ph, W + pw
            t = t.view(B, C, Hp // M, M, Wp // M, M).permute(0, 2, 4, 3, 5, 1).reshape(-1, M * M, C)
            q = self._heads(self.q(t), M * M)
            k, v = self.kv(t).chunk(2, dim=-1)
            o = self._softmax_attn(q, self._heads(k, M * M), self._heads(v, M * M))
            o = o.transpose(1, 2).reshape(B, Hp // M, Wp // M, M, M, self.d)
            o = o.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, self.d)[:, :H, :W]
            out = o.reshape(B, n, self.d)
        elif self.attention == "linear":
            q = F.elu(self._heads(self.q(tok), n)) + 1
            k, v = self.kv(tok).chunk(2, dim=-1)
            k, v = F.elu(self._heads(k, n)) + 1, self._heads(v, n)
            q, k, v = q.float(), k.float(), v.float()
            kv = k.transpose(-2, -1) @ v                      # (B, k, dk, dk)
            z = 1.0 / (q @ k.sum(dim=2, keepdim=True).transpose(-2, -1) + 1e-6)  # (B, k, n, 1)
            out = ((q @ kv) * z).to(tok.dtype).transpose(1, 2).reshape(B, n, self.d)
        else:
            raise ValueError(self.attention)
        out = self.proj(out)  # W^O
        return out.transpose(1, 2).reshape(B, C, H, W)


class DWSC(nn.Module):
    """Branch 1: 3x3 depth-wise separable convolution + BN + ReLU (Eq. 1)."""

    def __init__(self, c):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return F.relu(self.bn(self.pw(self.dw(x))))


# ============================================================================ DBSA
class DBSA(nn.Module):
    """Y = X + G * F_conv + (1 - G) * F_attn, with G = sigmoid(W2 ReLU(W1 GAP([F_conv;F_attn]) + b1) + b2)."""

    def __init__(self, channels: int, heads: int = 8, attention: str = "dense", fusion: str = "gate",
                 branches: str = "both", use_sdpa: bool = False):
        super().__init__()
        self.fusion, self.branches = fusion, branches
        self.conv = DWSC(channels) if branches in ("both", "conv") else None
        self.attn = MHSA(channels, heads, attention, use_sdpa=use_sdpa) if branches in ("both", "attn") else None
        if branches == "both":
            if fusion in ("gate", "gate_nores"):
                self.gate = nn.Sequential(nn.Linear(2 * channels, channels // 2), nn.ReLU(inplace=True),
                                          nn.Linear(channels // 2, channels))
            elif fusion == "concat":
                self.mix = nn.Conv2d(2 * channels, channels, 1)
        self.last_gate: Optional[torch.Tensor] = None
        self.last_update_ratio: Optional[torch.Tensor] = None
        self.record = False

    def forward(self, x):
        if self.branches == "conv":
            upd = self.conv(x)
        elif self.branches == "attn":
            upd = self.attn(x)
        else:
            fc, fa = self.conv(x), self.attn(x)
            if self.fusion in ("gate", "gate_nores"):
                pooled = torch.cat([fc, fa], 1).mean(dim=(2, 3))            # GAP([F_conv;F_attn])
                g = torch.sigmoid(self.gate(pooled))[:, :, None, None]      # (B, C, 1, 1)
                if self.record:
                    self.last_gate = g.detach().flatten(1)
                upd = g * fc + (1 - g) * fa
                if self.fusion == "gate_nores":
                    return upd
            elif self.fusion == "avg":
                upd = 0.5 * fc + 0.5 * fa
            elif self.fusion == "add":
                upd = fc + fa
            elif self.fusion == "concat":
                upd = self.mix(torch.cat([fc, fa], 1))
            else:
                raise ValueError(self.fusion)
        if self.record:
            self.last_update_ratio = (upd.detach().flatten(1).norm(dim=1) /
                                      x.detach().flatten(1).norm(dim=1).clamp_min(1e-8))
        return x + upd


# ============================================================================ replacement modules (ablation c)
class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 4), c))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean(dim=(2, 3))))[:, :, None, None]


class CBAM(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(c, max(c // r, 4)), nn.ReLU(inplace=True), nn.Linear(max(c // r, 4), c))
        self.spatial = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean(dim=(2, 3))) + self.mlp(x.amax(dim=(2, 3))))[:, :, None, None]
        x = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)))
        return x * sa


class MobileViTBlock(nn.Module):
    """MobileViT block (Mehta & Rastegari): local conv -> transformer on unfolded 2x2 patches -> fuse."""

    def __init__(self, c, d=None, depth=1, heads=4, patch=2):
        super().__init__()
        d = d or max(32, c)
        self.p = patch
        self.local = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU(),
                                   nn.Conv2d(c, d, 1, bias=False))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True,
                                           activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, depth)
        self.proj = nn.Sequential(nn.Conv2d(d, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU())

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.p
        y = self.local(x)
        ph, pw = (p - H % p) % p, (p - W % p) % p
        y = F.pad(y, (0, pw, 0, ph))
        Hp, Wp = H + ph, W + pw
        D = y.shape[1]
        y = y.view(B, D, Hp // p, p, Wp // p, p).permute(0, 3, 5, 2, 4, 1).reshape(B * p * p, -1, D)
        y = self.transformer(y)
        y = y.view(B, p, p, Hp // p, Wp // p, D).permute(0, 5, 3, 1, 4, 2).reshape(B, D, Hp, Wp)[:, :, :H, :W]
        y = self.proj(y)
        return self.fuse(torch.cat([x, y], 1))


# ============================================================================ HTA-ConvNet
STAGE_CHANNELS = {1: 24, 2: 32, 3: 56, 4: 160, 5: 1792}


class HTAConvNet(nn.Module):
    """EfficientNet-B4 grouped into 5 stages (Table 'backbone'); a block is inserted after selected stages.

    torchvision efficientnet_b4.features: [0] stem, [1] MBConv1 (24@112), [2] (32@56), [3] (56@28),
    [4] (112@14), [5] (160@14), [6] (272@7), [7] (448@7), [8] 1x1 conv (1792@7).
    """

    def __init__(self, num_classes=38, stages: Sequence[int] = (2, 3, 4), heads: int = 8,
                 attention: str = "dense", stage_attention: Optional[Dict[int, str]] = None,
                 fusion: str = "gate", branches: str = "both", module: str = "dbsa",
                 dropout: float = 0.4, pretrained: bool = True, use_sdpa: bool = False):
        super().__init__()
        weights = torchvision.models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.efficientnet_b4(weights=weights).features
        self.stages = nn.ModuleList([nn.Sequential(f[0], f[1]), nn.Sequential(f[2]), nn.Sequential(f[3]),
                                     nn.Sequential(f[4], f[5]), nn.Sequential(f[6], f[7], f[8])])
        self.blocks = nn.ModuleDict()
        stage_attention = stage_attention or {}
        for s in stages:
            c = STAGE_CHANNELS[s]
            if module == "dbsa":
                blk = DBSA(c, heads, stage_attention.get(s, attention), fusion, branches, use_sdpa)
            elif module == "se":
                blk = SEBlock(c)
            elif module == "cbam":
                blk = CBAM(c)
            elif module == "mobilevit":
                blk = MobileViTBlock(c)
            else:
                raise ValueError(module)
            self.blocks[str(s)] = blk
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(1792, num_classes)

    def features(self, x):
        for i, st in enumerate(self.stages, start=1):
            x = st(x)
            if str(i) in self.blocks:
                x = self.blocks[str(i)](x)
        return x

    def forward(self, x):
        return self.fc(self.dropout(torch.flatten(self.pool(self.features(x)), 1)))

    def dbsa_blocks(self) -> Dict[str, DBSA]:
        return {k: b for k, b in self.blocks.items() if isinstance(b, DBSA)}

    def set_record(self, flag: bool):
        for b in self.dbsa_blocks().values():
            b.record = flag


def build_hta(args_or_kwargs=None, **kw) -> HTAConvNet:
    return HTAConvNet(**kw)


# ============================================================================ analytical cost (Eqs. 10-12)
def analytical_dbsa_cost(img=224, heads=8, stages=(2, 3, 4), attention="dense", stage2_attention=None):
    """Returns per-stage (n, C, DWSC MFLOPs, MHSA MFLOPs, params K) following the manuscript formulas."""
    res = {2: img // 4, 3: img // 8, 4: img // 16}
    rows = []
    for s in stages:
        hw = res[s]
        n, C = hw * hw, STAGE_CHANNELS[s]
        d = heads * math.ceil(C / heads)
        att = stage2_attention if (s == 2 and stage2_attention) else attention
        dwsc = n * C * (9 + C)
        proj = 4 * n * C * d
        if att == "dense":
            core = 2 * n * n * d
        elif att == "window":
            core = 2 * n * 49 * d
        elif att.startswith("sra"):
            r = int(att[3:])
            m = (hw // r) ** 2
            proj = 2 * n * C * d + 2 * m * C * d          # Q, out on n tokens; K, V on m tokens
            core = 2 * n * m * d + m * C * C * r * r      # attention + RxR reduction conv
        elif att == "linear":
            core = 2 * n * d * d // heads
        mhsa = proj + core
        params = (C * C + 11 * C) + (4 * C * d + 3 * d + C) + 2 * C + (1.5 * C * C + 1.5 * C)
        rows.append(dict(stage=s, n=n, C=C, attention=att, dwsc_mflops=dwsc / 1e6, mhsa_mflops=mhsa / 1e6,
                         params_k=params / 1e3, attn_map_mb_fp16=(n * n * heads * 2 / 1e6 if att == "dense" else
                                                                  n * 49 * heads * 2 / 1e6 if att == "window" else
                                                                  n * ((hw // int(att[3:])) ** 2) * heads * 2 / 1e6
                                                                  if att.startswith("sra") else 0.0)))
    return rows
