"""Masked Autoencoder with a ViT-Tiny/16 encoder.

Follows He et al. (arXiv:2111.06377), with one deliberate deviation:

    THE MASK IS AN INPUT, NOT GENERATED INTERNALLY.

Upstream MAE derives the mask from sorted noise inside `forward_encoder`. Here every masking
policy lives in src/masking/ and the model just consumes a boolean mask. That is what lets the
experimental arms share one model, one optimiser and one loss.

Transformer blocks are hand-written rather than imported from timm: the code stays readable
and attention weights stay directly accessible for the attention-distance measurement.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos_embed import get_2d_sincos_pos_embed, patch_centres


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, need_attn: bool = False):
        n, t, c = x.shape
        qkv = (self.qkv(x)
               .reshape(n, t, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        if need_attn:  # explicit path: the attention-distance measurement needs the weights
            attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
            x = attn @ v
        else:          # fused kernel for everything else
            attn = None
            x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(n, t, c)), attn


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, need_attn: bool = False):
        h, attn = self.attn(self.norm1(x), need_attn)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, attn


class MaskedAutoencoderViT(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 192, depth: int = 12, num_heads: int = 3,
                 decoder_embed_dim: int = 128, decoder_depth: int = 4,
                 decoder_num_heads: int = 4, mlp_ratio: float = 4.0,
                 norm_pix_loss: bool = True) -> None:
        super().__init__()
        if img_size % patch_size:
            raise ValueError("img_size must be divisible by patch_size")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.norm_pix_loss = norm_pix_loss

        # --- encoder ---
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim),
                                      requires_grad=False)
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # --- decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList(
            [Block(decoder_embed_dim, decoder_num_heads, mlp_ratio)
             for _ in range(decoder_depth)])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

        self.initialize_weights()

    # ---------------------------------------------------------------------------------
    def initialize_weights(self) -> None:
        for attr, dim in (("pos_embed", self.pos_embed.shape[-1]),
                          ("decoder_pos_embed", self.decoder_pos_embed.shape[-1])):
            pe = get_2d_sincos_pos_embed(dim, self.grid_size, cls_token=True)
            getattr(self, attr).data.copy_(torch.from_numpy(pe).float().unsqueeze(0))

        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, L, patch**2 * 3)."""
        p, g = self.patch_size, self.grid_size
        x = imgs.reshape(imgs.shape[0], self.in_chans, g, p, g, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], g * g, p * p * self.in_chans)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """(N, L, patch**2 * 3) -> (N, 3, H, W)."""
        p, g = self.patch_size, self.grid_size
        x = x.reshape(x.shape[0], g, g, p, p, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_chans, g * p, g * p)

    # ---------------------------------------------------------------------------------
    def forward_encoder(self, imgs: torch.Tensor, mask: torch.Tensor):
        """`mask`: bool (N, L), True = masked/removed. Every row must mask the same count.

        Returns (latent (N, 1 + L - M, D), ids_restore (N, L)).
        """
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)  # (N, L, D)
        x = x + self.pos_embed[:, 1:, :]                       # pos-embed BEFORE gather
        n, ell, d = x.shape

        mask = mask.to(torch.bool)
        counts = mask.sum(dim=1)
        m = int(counts[0].item())
        if not bool((counts == m).all()):
            raise AssertionError(
                f"budget parity violated: mask counts range "
                f"[{int(counts.min())}, {int(counts.max())}]; every row must mask "
                "exactly M patches or the batched gather is ill-defined"
            )

        # Sort so kept (False) tokens come first. Tiny noise randomises tie order; the
        # encoder is permutation-invariant here because pos-embed is already added.
        noise = torch.rand(n, ell, device=x.device) * 1e-3
        ids_shuffle = torch.argsort(mask.float() + noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, : ell - m]

        x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)

        for blk in self.blocks:
            x, _ = blk(x)
        return self.norm(x), ids_restore

    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        n, ell = ids_restore.shape
        d = x.shape[2]

        mask_tokens = self.mask_token.expand(n, ell + 1 - x.shape[1], -1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, d))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x, _ = blk(x)
        return self.decoder_pred(self.decoder_norm(x))[:, 1:, :]

    def forward_loss(self, imgs: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        """Returns (scalar loss over masked patches, per-patch loss (N, L)).

        Normalised-pixel targets follow the MAE recipe. They also make per-category loss
        comparable: without per-patch brightness and contrast, a "hard category" is hard
        rather than merely high-variance.
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6).sqrt()

        per_patch = (pred.float() - target.float()).pow(2).mean(dim=-1)
        m = mask.to(per_patch.dtype)
        loss = (per_patch * m).sum() / m.sum().clamp_min(1.0)
        return loss, per_patch

    def forward(self, imgs: torch.Tensor, mask: torch.Tensor):
        latent, ids_restore = self.forward_encoder(imgs, mask)
        pred = self.forward_decoder(latent, ids_restore)
        loss, per_patch = self.forward_loss(imgs, pred, mask)
        return loss, pred, per_patch

    # ---------------------------------------------------------------------------------
    # paths used after pretraining
    # ---------------------------------------------------------------------------------
    def tokens(self, imgs: torch.Tensor, last_k: int = 4) -> torch.Tensor:
        """Patch tokens on the FULL token set (no masking), last `last_k` blocks concatenated.

        Returns (N, L, D * last_k). A single 192-d vector is a thin basis for an 80-way task;
        concatenating four blocks gives 768-d while any head on top stays just as simple.
        No no_grad here, because fine-tuning backpropagates through this path.
        """
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:, :]
        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)

        out = []
        for i, blk in enumerate(self.blocks):
            x, _ = blk(x)
            if i >= len(self.blocks) - last_k:
                out.append(x[:, 1:, :])
        return torch.cat(out, dim=-1)

    @torch.no_grad()
    def attention_distance(self, imgs: torch.Tensor, mask: torch.Tensor | None = None
                           ) -> torch.Tensor:
        """Mean attention distance per layer, in pixels.

        `mask=None` uses all L tokens, the standard inference protocol. Passing a frozen eval
        mask instead matches the pretraining sequence length. Both are reported, so the
        conclusion does not depend on the protocol.

        Returns a (depth,) tensor of mean distances.
        """
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:, :]
        n, ell, d = x.shape
        centres = torch.from_numpy(patch_centres(self.grid_size, self.patch_size)).to(x.device)

        if mask is not None:
            mask = mask.to(torch.bool)
            m = int(mask.sum(dim=1)[0].item())
            noise = torch.rand(n, ell, device=x.device) * 1e-3
            ids_shuffle = torch.argsort(mask.float() + noise, dim=1)
            ids_keep = ids_shuffle[:, : ell - m]
            x = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, d))
            coords = centres[ids_keep]                       # (N, T, 2)
        else:
            coords = centres.unsqueeze(0).expand(n, -1, -1)  # (N, L, 2)

        cls = (self.cls_token + self.pos_embed[:, :1, :]).expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # pairwise patch-centre distances, CLS excluded from both axes
        dist = torch.cdist(coords, coords)                   # (N, T, T)
        out = []
        for blk in self.blocks:
            x, attn = blk(x, need_attn=True)
            a = attn[:, :, 1:, 1:].float()                   # drop CLS row/col
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            out.append((a * dist.unsqueeze(1)).sum(-1).mean().item())
        return torch.tensor(out)


def build_mae(cfg) -> MaskedAutoencoderViT:
    """The one place that turns a Config into a model, so every stage builds the same one."""
    return MaskedAutoencoderViT(img_size=cfg.data.img_size, patch_size=cfg.data.patch_size,
                                **vars(cfg.model))


def count_params(model: nn.Module) -> tuple[int, int]:
    """(encoder params, total trainable params). Fixed pos-embeds are excluded."""
    enc_prefixes = ("patch_embed", "cls_token", "blocks", "norm.")
    enc = sum(p.numel() for n, p in model.named_parameters()
              if p.requires_grad and n.startswith(enc_prefixes))
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return enc, total
