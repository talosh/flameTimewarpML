# warpnet4_v002 — refwarp (bidirectional reference-guided warping) with a
# frozen ts_test ViT latent-encoder side branch.
#
# = warpnet4_v001 (unchanged subset: hpass, conv, warp, Head, ScalarEmbed,
#   FlownetFiLM with sharpness film + scaler, FlownetCas) plus:
#
#   * `ae_encoder`  : the ts_test ViT LatentEncoder (6 blocks, dim 256,
#                     6.05M params, 110 tensors), FROZEN by default.
#                     Provenance: /mnt/StorageMedia/dev/ts_test/src/models/
#                     vit_latent_ae.py (LATENT_VERSION "1"); trained weights
#                     from /mnt/StorageMedia/dev/ts_test/runs/m6_full/
#                     ckpt_final.pt ['encoder'] (59152 steps, 1 epoch over
#                     pexels_exr). tools/merge_warpnet_v002.py bakes them
#                     into the v002 checkpoint and verifies key-for-key
#                     parity with the ts_test class (110/110 keys + shapes,
#                     checked 2026-09-07).
#
#   * `film_z`      : zero-init latent FiLM (Linear(256 -> 2c)) on each
#                     ResConv, applied to the conv branch AFTER the
#                     sharpness film:
#                       h   = conv(x)
#                       h   = h * (1 + g) + s      (sharpness film, v001)
#                       h   = h * (1 + gz) + sz    (latent film, v002 new)
#                       out = PReLU(h * beta + x)
#
# PASSTHROUGH (why the current warpnet4_v001 weights load as-is):
#   film_z is zero-init, so at load the latent modulates nothing and the
#   block computes exactly the v001 function — bit-for-bit, for ANY
#   ae_encoder weights (the latent is multiplied by a zero matrix). The
#   merge script verifies it: v001(base) vs v002(merged) on a fixed-seed
#   input must give max|Δ| == 0.0 over all 4 flow levels.
#
# AE INPUT CONTRACT (canonical implementation lives HERE; the trainer and
# the node feed it — do not re-implement the pipeline elsewhere):
#   ae_in    : (a, b), each [B, 3, H, W] float32, AP1 (ACEScg) LINEAR,
#              arbitrary size, a.shape == b.shape. For the refwarp task
#              this is the (image, reference) pair, same order as the main
#              net.
#   encoding : 'ap1' | 'ap0' | 'rec709' | 'acescct' (same for both frames;
#              the per-step encoding the main net is seeing).
#   pipeline : 1. sanitize   nan_to_num(nan=0, posinf=AE_LIN_HI, neginf=0)
#              2. soft-limit to the OPEN interval (0, AE_LIN_HI), knee
#                 AE_LIN_KNEE (user decision 2026-09-07: bounded linear
#                 source, gradient survives at the rails instead of dying
#                 at a hard clamp). Deliberate deviation from the AE's
#                 raw-linear training contract (no upper bound):
#                 highlights above AE_LIN_HI roll off to the rail instead
#                 of running raw through tanh/OETF/cct.
#              3. resize     short side -> 64, aspect preserved (area
#                 downscale / bilinear upscale), in the LINEAR domain
#                 (ts_test contract: resampling in log/gamma space is a
#                 common silent error)
#              4. encode     to `encoding` (constants vendored from ts_test,
#                 canonical: src/data/latent_ae_dataset.py)
#              5. normalize  per-encoding to ~[-1, 1] (the AE's trained
#                 COLOUR_NORM contract — hard clamps unchanged)
#              6. LatentEncoder(a, b, ratio=AE_RATIO_FIXED) -> z_global
#                 [B, 256]
#   ratio    : fixed 0.5 — refwarp has no temporal ratio; 0.5 is
#              in-distribution (width-3 windows always yield 0.5), so the
#              latent's ratio dependence acts as a constant offset.
#   ae_in None: the latent path is off; film_z(0) = its bias (zero at
#              init), so the model computes exactly the v001 function. The
#              node must pass ae_in to enable the latent at inference.
#
# Note: the v001 file's unused `Flownet` (non-FiLM) class is dropped here;
# only FlownetFiLM is instantiated by FlownetCas.

import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:

    info = {
        'name': 'Warpnet4_v002',
        'file': 'warpnet4_v002.py',
        'ratio_support': True,
        'ae_latent_version': '1'
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        backwarp_tenGrid = {}
        gauss_cache = {}

        # -------------------------------------------------------------------
        # AE contract constants (module docstring)
        # -------------------------------------------------------------------
        AE_LATENT_VERSION = '1'
        AE_PATCH = 8
        AE_DIM = 256
        AE_N_HEADS = 4
        AE_DEPTH = 6
        AE_MLP_RATIO = 4
        AE_MAX_GRID = 512
        AE_N_RATIO_FREQ = 32
        AE_SHORT_SIDE = 64
        AE_RATIO_FIXED = 0.5
        AE_LIN_LO = 0.0
        AE_LIN_HI = 1.1          # '0 - 1 or maybe 0 - 1.1' (user, 2026-09-07)
        AE_LIN_KNEE = 0.002
        AE_ENCODINGS = ('ap1', 'ap0', 'rec709', 'acescct')

        # -------------------------------------------------------------------
        # Colour constants (vendored; canonical: ts_test/src/data/
        # latent_ae_dataset.py — do not edit independently). float64 source
        # values rounded to float32, exactly as the ts_test pipeline does.
        # -------------------------------------------------------------------
        AP1_TO_AP0 = torch.tensor([
            [ 0.6954522414,  0.1406786965,  0.1638690622],
            [ 0.0447945634,  0.8596711185,  0.0955343182],
            [-0.0055258826,  0.0040252103,  1.0015006723],
        ], dtype=torch.float64).float()
        AP1_TO_REC709 = torch.tensor([
            [ 1.7312716068, -0.6040492953, -0.0801085107],
            [-0.1316162926,  1.1348188172, -0.0086792643],
            [-0.0245723770, -0.1257715177,  1.0658158680],
        ], dtype=torch.float64).float()

        OETF_709_K = 4.5
        OETF_709_X0 = 0.018
        OETF_709_A = 1.099
        OETF_709_P = 1.0 / 2.4
        OETF_709_B = 0.099

        CCT_A = 10.5402377416545
        CCT_B = 0.0729055341958355
        CCT_X_BRK = 0.0078125
        CCT_LOG_NUM = 9.72
        CCT_LOG_DEN = 17.52
        CCT_LIN_MAX = 65504.0
        CCT_F0 = CCT_B
        CCT_FMAX = (math.log2(CCT_LIN_MAX) + CCT_LOG_NUM) / CCT_LOG_DEN

        # -------------------------------------------------------------------
        # AE input pipeline (module docstring, steps 1-5)
        # -------------------------------------------------------------------
        def soft_limit(x, lo=AE_LIN_LO, hi=AE_LIN_HI, knee=AE_LIN_KNEE):
            # User-provided (2026-09-07). Holds the output inside the open
            # interval (lo, hi); identity inside [lo+knee, hi-knee]; the
            # roll-off keeps gradient alive at the rails (slope
            # 1/(1+t)^2, never the dead rail of a hard clamp). Inputs are
            # pre-sanitized (finite, within [lo, hi] +- a bit) so t stays
            # small and knee*t cannot overflow float32.
            top = hi - knee
            t = (x - top) / knee
            x = torch.where(x > top, top + knee * t / (1.0 + t.clamp(min=0.0)), x)
            bot = lo + knee
            t = (bot - x) / knee
            x = torch.where(x < bot, bot - knee * t / (1.0 + t.clamp(min=0.0)), x)
            return x

        def _ae_bt709_oetf(x):
            # BT.709 OETF (ITU-R BT.709) on a non-negative linear tensor
            x = x.clamp_min(0.0)
            return torch.where(x <= OETF_709_X0,
                               OETF_709_K * x,
                               OETF_709_A * x.pow(OETF_709_P) - OETF_709_B)

        def _ae_acescct(x):
            # ACEScct (ACES spec S-2016-001) on non-negative lin_AP1
            x = x.clamp_min(0.0)
            log_part = (x.clamp_min(torch.finfo(x.dtype).tiny).log2()
                        + CCT_LOG_NUM) / CCT_LOG_DEN
            return torch.where(x <= CCT_X_BRK, CCT_A * x + CCT_B, log_part)

        def _ae_encode_linear(x, name):
            # ts_test encode_linear, batched: [B, 3, H, W] linear AP1
            # (sanitized + soft-limited) -> [B, 3, H, W] in [-1, 1]
            if name == 'ap1':
                return torch.tanh(x)
            if name == 'ap0':
                return torch.tanh(torch.einsum('ij,bjhw->bihw', AP1_TO_AP0, x))
            if name == 'rec709':
                lin = torch.einsum('ij,bjhw->bihw', AP1_TO_REC709, x)
                return torch.clamp(2.0 * _ae_bt709_oetf(lin) - 1.0, -1.0, 1.0)
            if name == 'acescct':
                y = _ae_acescct(x)
                return torch.clamp(2.0 * (y - CCT_F0) / (CCT_FMAX - CCT_F0) - 1.0,
                                   -1.0, 1.0)
            raise ValueError(f'unknown encoding {name!r}; expected one of {AE_ENCODINGS}')

        def ae_prepare(a, b, encoding):
            # Contract steps 1-5 per frame (docstring). a, b: [B, 3, H, W]
            # AP1-linear; returns encoded+normalized [B, 3, 64, L].
            def prep(x):
                h, w = x.shape[-2:]
                short, long_ = min(h, w), max(h, w)
                new_long = max(AE_SHORT_SIDE, int(round(long_ * (AE_SHORT_SIDE / short))))
                out_h, out_w = (AE_SHORT_SIDE, new_long) if h <= w else (new_long, AE_SHORT_SIDE)
                if (out_h, out_w) != (h, w):
                    if out_h * out_w <= h * w:
                        x = F.interpolate(x, size=(out_h, out_w), mode='area')
                    else:
                        x = F.interpolate(x, size=(out_h, out_w), mode='bilinear', align_corners=False)
                return _ae_encode_linear(x, encoding)
            return prep(a), prep(b)

        # -------------------------------------------------------------------
        # ts_test ViT LatentEncoder (vendored verbatim; parameter names are
        # the state-dict contract with runs/m6_full/ckpt_final.pt ['encoder'])
        # -------------------------------------------------------------------
        def _ae_pad_to_patch_grid(x, patch=AE_PATCH):
            _, _, h, w = x.shape
            hp = (h + patch - 1) // patch * patch
            wp = (w + patch - 1) // patch * patch
            if hp != h or wp != w:
                x = F.pad(x, (0, wp - w, 0, hp - h))
            return x, (hp, wp)

        def _ae_sinusoidal_ratio(ratio, n_freq=AE_N_RATIO_FREQ):
            freqs = torch.pow(2.0, torch.arange(n_freq, device=ratio.device, dtype=ratio.dtype))
            angles = 2.0 * math.pi * ratio.unsqueeze(-1) * freqs
            return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        class _AeAttention(nn.Module):
            def __init__(self, dim, n_heads):
                super().__init__()
                if dim % n_heads != 0:
                    raise ValueError(f'dim {dim} not divisible by n_heads {n_heads}')
                self.n_heads = n_heads
                self.head_dim = dim // n_heads
                self.qkv = nn.Linear(dim, 3 * dim)
                self.proj = nn.Linear(dim, dim)

            def forward(self, x):
                b, n, c = x.shape
                qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                x = F.scaled_dot_product_attention(q, k, v)
                return self.proj(x.transpose(1, 2).reshape(b, n, c))

        class _AeBlock(nn.Module):
            # Pre-norm transformer block: x + attn(LN(x)); x + mlp(LN(x))
            def __init__(self, dim, n_heads, mlp_ratio):
                super().__init__()
                self.norm1 = nn.LayerNorm(dim)
                self.attn = _AeAttention(dim, n_heads)
                self.norm2 = nn.LayerNorm(dim)
                self.mlp = nn.Sequential(
                    nn.Linear(dim, dim * mlp_ratio),
                    nn.GELU(),
                    nn.Linear(dim * mlp_ratio, dim),
                )

            def forward(self, x):
                x = x + self.attn(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x

        class LatentEncoder(nn.Module):
            """(frame_a, frame_b, ratio) -> (z_tokens, z_global).

            Standalone artifact of ts_test — vendored verbatim (see
            module docstring for provenance). z_global [B, 256] is what
            film_z consumes; z_tokens [B, T, 256] is returned for parity
            with the ts_test contract (spatial conditioning = v2 idea).
            """
            def __init__(self, dim=AE_DIM, n_heads=AE_N_HEADS, depth=AE_DEPTH,
                         mlp_ratio=AE_MLP_RATIO, patch=AE_PATCH, max_grid=AE_MAX_GRID,
                         n_ratio_freq=AE_N_RATIO_FREQ):
                super().__init__()
                self.dim = dim
                self.patch = patch
                self.max_grid = max_grid
                self.depth = depth

                self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
                self.emb_type_a = nn.Parameter(torch.zeros(dim))
                self.emb_type_b = nn.Parameter(torch.zeros(dim))
                self.pos_row = nn.Parameter(torch.zeros(max_grid, dim))
                self.pos_col = nn.Parameter(torch.zeros(max_grid, dim))

                self.ratio_mlp = nn.Sequential(
                    nn.Linear(2 * n_ratio_freq, dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                )
                # Per-block FiLM inside the AE (trained, NOT zero at ckpt):
                self.film_gamma = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
                self.film_beta = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))
                for m in list(self.film_gamma) + list(self.film_beta):
                    nn.init.zeros_(m.weight)
                    nn.init.zeros_(m.bias)

                self.blocks = nn.ModuleList(
                    _AeBlock(dim, n_heads, mlp_ratio) for _ in range(depth)
                )
                self.norm_out = nn.LayerNorm(dim)
                self.latent_proj = nn.Linear(2 * dim, dim)

            def _positions(self, gh, gw):
                if gh > self.max_grid or gw > self.max_grid:
                    raise ValueError(
                        f'grid {gh}x{gw} exceeds max_grid={self.max_grid} patches/axis')
                pos = self.pos_row[:gh, None, :] + self.pos_col[None, :gw, :]
                return pos.reshape(gh * gw, self.dim)

            def forward(self, a, b, ratio):
                if a.shape != b.shape:
                    raise ValueError(
                        f'frame_a {tuple(a.shape)} and frame_b {tuple(b.shape)} must match')
                bsz, _, h, w = a.shape
                a, (hp, wp) = _ae_pad_to_patch_grid(a, self.patch)
                b, _ = _ae_pad_to_patch_grid(b, self.patch)
                gh, gw = hp // self.patch, wp // self.patch
                t = gh * gw

                x_a = self.patch_embed(a).flatten(2).transpose(1, 2)
                x_b = self.patch_embed(b).flatten(2).transpose(1, 2)
                pos = self._positions(gh, gw)
                x_a = x_a + self.emb_type_a + pos
                x_b = x_b + self.emb_type_b + pos

                r = self.ratio_mlp(_ae_sinusoidal_ratio(ratio))
                x = torch.cat([r.unsqueeze(1), x_a, x_b], dim=1)

                for i, blk in enumerate(self.blocks):
                    gamma = self.film_gamma[i](r).unsqueeze(1)
                    beta = self.film_beta[i](r).unsqueeze(1)
                    x = torch.cat([x[:, :1], x[:, 1:] * (1.0 + gamma) + beta], dim=1)
                    x = blk(x)
                x = self.norm_out(x)

                body = x[:, 1:].reshape(bsz, 2, t, self.dim)
                z_tokens = self.latent_proj(body.reshape(bsz, t, 2 * self.dim))
                z_global = z_tokens.mean(dim=1)
                return z_tokens, z_global

        # -------------------------------------------------------------------
        # warpnet4_v001 components (unchanged)
        # -------------------------------------------------------------------
        def hpass(img):
            def gauss_kernel(channels):
                kernel = torch.tensor([[1., 4., 6., 4., 1.],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]])
                kernel /= 256.
                return kernel.repeat(channels, 1, 1, 1)

            def conv_gauss(img, kernel):
                img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
                return torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])

            k = (img.shape[1], str(img.device), str(img.dtype))
            if k not in gauss_cache:
                gauss_cache[k] = gauss_kernel(img.shape[1]).to(device=img.device, dtype=img.dtype)
            gkernel = gauss_cache[k]

            hp = img - conv_gauss(img, gkernel)

            hp = hp + 0.5

            return hp

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    padding_mode = 'reflect',
                    bias=True
                ),
                torch.nn.PReLU(num_parameters=out_planes, init=0.2)
            )

        def warp(tenInput, tenFlow):
            k = (str(tenFlow.device), str(tenFlow.dtype), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.PReLU(num_parameters=32, init=0.2)

                self.maxdepth = 2

            def forward(self, x, feat=False):
                n, c, h, w = x.shape
                ph = (-h) % self.maxdepth
                pw = (-w) % self.maxdepth
                x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode='replicate')

                x = x + 2.0 * (hpass(x)) - 1.0

                x0 = self.cnn0(x)
                x = self.relu(x0)
                x1 = self.cnn1(x)
                x = self.relu(x1)
                x2 = self.cnn2(x)
                x = self.relu(x2)
                x3 = self.cnn3(x)
                if feat:
                    return [x0, x1, x2, x3]
                return x3[:, :, :h, :w]

        class ScalarEmbed(nn.Module):
            def __init__(self, dim=128, n_freqs=16, max_log2=6.0):
                super().__init__()
                self.register_buffer(
                    "freqs", math.pi * 2.0 ** torch.linspace(0.0, max_log2, n_freqs)
                )
                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(2 * n_freqs + 1, dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(dim, dim)
                )

            def forward(self, t, n=1, device=None, dtype=None):
                if not torch.is_tensor(t):
                    t = torch.tensor(t, device=device, dtype=dtype)
                t = t.reshape(-1).to(device=device, dtype=dtype)
                if t.numel() == 1 and n > 1:
                    t = t.expand(n)
                a = t[:, None] * self.freqs[None, :].to(dtype=t.dtype)
                return self.mlp(
                    torch.cat([t[:, None], a.sin(), a.cos()], 1)
                )

        # -------------------------------------------------------------------
        # ResConv: the v001 block plus the zero-init latent FiLM (film_z)
        # -------------------------------------------------------------------
        class ResConv(Module):
            def __init__(self, c, dilation=1, cond_dim=128, z_dim=AE_DIM):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(num_parameters=c, init=0.2)

                # sharpness conditioning (v001) — zero-init => passthrough
                self.film = torch.nn.Linear(cond_dim, 2 * c)
                torch.nn.init.zeros_(self.film.weight)
                torch.nn.init.zeros_(self.film.bias)
                # latent conditioning (v002) — zero-init => passthrough
                self.film_z = torch.nn.Linear(z_dim, 2 * c)
                torch.nn.init.zeros_(self.film_z.weight)
                torch.nn.init.zeros_(self.film_z.bias)

            def forward(self, inp):
                x, emb, z = inp
                h = self.conv(x)
                gamma, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
                h = h * (1 + gamma) + shift
                g_z, s_z = self.film_z(z)[:, :, None, None].chunk(2, dim=1)
                h = h * (1 + g_z) + s_z
                return self.relu(h * self.beta + x), emb, z

        class FlownetFiLM(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 3, 2, 1),
                    conv(c//2, c, 3, 2, 1),
                    )
                self.convblock = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )

                self.scaler = torch.nn.Conv2d(c, c, 1, groups=c)
                torch.nn.init.ones_(self.scaler.weight)
                torch.nn.init.zeros_(self.scaler.bias)

                self.emb = ScalarEmbed()

                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1, cnd=0, z=None):
                n, _, h, w = img0.shape

                sh = max(1, round(h / scale))
                sw = max(1, round(w / scale))
                ph = (-sh) % self.maxdepth
                pw = (-sw) % self.maxdepth

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((img0, img1, f0, f1, timestep), 1)
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=True)

                if flow is not None:
                    mask_s = torch.nn.functional.interpolate(mask, size=(sh, sw), mode="bilinear", align_corners=True)
                    flow_s = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=True)
                    down = flow_s.new_tensor([sw / w, sh / h, sw / w, sh / h]).view(1, 4, 1, 1)
                    x = torch.cat((x, mask_s, flow_s * down), 1)

                emb = self.emb(cnd, n=n, device=x.device, dtype=x.dtype)
                if z is None:
                    # latent path off: film_z(0) = its bias (zero at init)
                    z = x.new_zeros(n, AE_DIM)

                x = torch.nn.functional.pad(x, (0, pw, 0, ph))

                feat = self.conv0(x)
                feat = self.scaler(feat)
                feat, _, _ = self.convblock((feat, emb, z))
                tmp = self.lastconv(feat)

                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=True)

                up = tmp.new_tensor([w / sw, h / sh, w / sw, h / sh]).view(1, 4, 1, 1)
                flow = tmp[:, :4] * up
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetFiLM(7+16, c=192)
                self.block1 = FlownetFiLM(8+4+16, c=128)
                self.block2 = FlownetFiLM(8+4+16, c=96)
                self.block3 = FlownetFiLM(8+4+16, c=64)
                self.encode = Head()
                self.ae_encoder = LatentEncoder()
                for p in self.ae_encoder.parameters():
                    p.requires_grad = False   # frozen by default (phase 3 unfreezes)

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], sharpness=0,
                        ae_in=None, encoding='ap1'):

                timestep=0.5

                def halving_steps(start, n_steps = 4):
                    vals = [max(start // (2 ** i), 1) for i in range(n_steps - 1)]
                    vals.append(1)
                    return vals

                def skewed_randint(low=1, high=16, alpha=1.0, beta=2.0):
                    x = random.betavariate(alpha, beta)
                    return low + min(int(x * (high - low + 1)), high - low)

                scale = halving_steps(skewed_randint(), 4)

                img0 = torch.nan_to_num(img0, nan=0.0, posinf=AE_LIN_HI, neginf=AE_LIN_LO)
                img0 = soft_limit(img0)
                img1 = torch.nan_to_num(img1, nan=0.0, posinf=AE_LIN_HI, neginf=AE_LIN_LO)
                img1 = soft_limit(img1)

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                # AE side branch (frozen): z_global [B, 256] (docstring
                # contract), or None when ae_in is None (latent off)
                if ae_in is not None:
                    a, b = ae_in
                    if a.shape != b.shape:
                        raise ValueError(
                            f'ae_in pair shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}')
                    a_enc, b_enc = ae_prepare(a, b, encoding)
                    ratio = torch.full((a.shape[0],), AE_RATIO_FIXED,
                                       device=a.device, dtype=a.dtype)
                    z = self.ae_encoder(a_enc, b_enc, ratio)[1]
                else:
                    z = None

                flow_list = [None] * 4
                mask_list = [None] * 4
                merged = [None] * 4

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0], cnd=sharpness, z=z)

                flow_list[0] = flow[:, [2, 3, 0, 1], :, :].clone()

                flow_d, mask, conf = self.block1(
                    warp(img0, flow[:, :2]),
                    warp(img1, flow[:, 2:4]),
                    warp(f0, flow[:, :2]),
                    warp(f1, flow[:, 2:4]),
                    timestep,
                    mask,
                    flow,
                    scale=scale[1],
                    cnd = sharpness,
                    z = z
                )
                flow = flow + flow_d

                flow_list[1] = flow[:, [2, 3, 0, 1], :, :].clone()

                flow_d, mask, conf = self.block2(
                    warp(img0, flow[:, :2]),
                    warp(img1, flow[:, 2:4]),
                    warp(f0, flow[:, :2]),
                    warp(f1, flow[:, 2:4]),
                    timestep,
                    mask,
                    flow,
                    scale=scale[2],
                    cnd = sharpness,
                    z = z
                )
                flow = flow + flow_d

                flow_list[2] = flow[:, [2, 3, 0, 1], :, :].clone()

                flow_d, mask, conf = self.block3(
                    warp(img0, flow[:, :2]),
                    warp(img1, flow[:, 2:4]),
                    warp(f0, flow[:, :2]),
                    warp(f1, flow[:, 2:4]),
                    timestep,
                    mask,
                    flow,
                    scale=scale[3],
                    cnd = sharpness,
                    z = z
                )
                flow = flow + flow_d

                flow_list[3] = flow[:, [2, 3, 0, 1], :, :]

                result = {
                    'flow_list': flow_list,
                    'scale': scale
                }

                return result

        self.model = FlownetCas
        self.training_model = FlownetCas

    @staticmethod
    def get_info():
        return Model.info

    @staticmethod
    def get_name():
        return Model.info.get('name')

    @staticmethod
    def input_channels(model_state_dict):
        channels = 3
        try:
            channels = model_state_dict.get('multiresblock1.conv_3x3.conv1.weight').shape[1]
        except Exception as e:
            print (f'Unable to get model dict input channels - setting to 3, {e}')
        return channels

    @staticmethod
    def output_channels(model_state_dict):
        channels = 5
        try:
            channels = model_state_dict.get('conv_final.conv1.weight').shape[0]
        except Exception as e:
            print (f'Unable to get model dict output channels - setting to 3, {e}')
        return channels

    def get_model(self):
        import platform
        if platform.system() == 'Darwin':
            return self.training_model
        return self.model

    def get_training_model(self):
        return self.training_model

    def load_model(self, path, flownet, rank=0):
        import torch
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                incompatible = flownet.load_state_dict(convert(torch.load(path)), False)
            else:
                incompatible = flownet.load_state_dict(convert(torch.load(path, map_location ='cpu')), False)

            missing = list(getattr(incompatible, 'missing_keys', []))
            unexpected = list(getattr(incompatible, 'unexpected_keys', []))
            if missing:
                print (f'load_model: {len(missing)} missing keys (kept at init), e.g. {missing[:6]}')
            if unexpected:
                print (f'load_model: {len(unexpected)} unexpected keys (ignored), e.g. {unexpected[:6]}')
