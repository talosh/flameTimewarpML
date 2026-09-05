import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model:

    info = {
        'name': 'Warpnet4_v001',
        'file': 'Warpnet4_v001.py',
        'ratio_support': True
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        backwarp_tenGrid = {}
        gauss_cache = {}

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

        class ResConv(Module):
            # identical to the pretrained block, plus a zero-init FiLM modulation
            # of the conv branch: gamma = beta = 0 at load -> exactly the old function
            def __init__(self, c, dilation=1, cond_dim=128):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(num_parameters=c, init=0.2)

                self.film = torch.nn.Linear(cond_dim, 2 * c)
                torch.nn.init.zeros_(self.film.weight)
                torch.nn.init.zeros_(self.film.bias)

            def forward(self, inp):
                x, emb = inp
                h = self.conv(x)
                gamma, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
                h = h * (1 + gamma) + shift
                return self.relu(h * self.beta + x), emb

        class Flownet(Module):
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

                self.emb = ScalarEmbed()

                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1, cnd=0):
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

                x = torch.nn.functional.pad(x, (0, pw, 0, ph))

                feat = self.conv0(x)
                feat, _ = self.convblock((feat, emb))
                tmp = self.lastconv(feat)

                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=True)

                up = tmp.new_tensor([w / sw, h / sh, w / sw, h / sh]).view(1, 4, 1, 1)
                flow = tmp[:, :4] * up
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

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

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1, cnd=0):
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

                x = torch.nn.functional.pad(x, (0, pw, 0, ph))

                feat = self.conv0(x)
                feat = self.scaler(feat)
                feat, _ = self.convblock((feat, emb))
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

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], sharpness=0):

                timestep=0.5

                def halving_steps(start, n_steps = 4):
                    vals = [max(start // (2 ** i), 1) for i in range(n_steps - 1)]
                    vals.append(1)
                    return vals

                def skewed_randint(low=1, high=16, alpha=1.0, beta=2.0):
                    x = random.betavariate(alpha, beta)
                    return low + min(int(x * (high - low + 1)), high - low)

                scale = halving_steps(skewed_randint(), 4)

                img0 = img0
                img1 = img1
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                merged = [None] * 4

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0], cnd=sharpness)

                flow_list[0] = flow[:, [2, 3, 0, 1], :, :].clone()
                # mask_list[0] = torch.sigmoid(mask.clone())
                # merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                flow_d, mask, conf = self.block1(
                    warp(img0, flow[:, :2]),
                    warp(img1, flow[:, 2:4]),
                    warp(f0, flow[:, :2]),
                    warp(f1, flow[:, 2:4]),
                    timestep,
                    mask,
                    flow,
                    scale=scale[1],
                    cnd = sharpness
                )
                flow = flow + flow_d

                flow_list[1] = flow[:, [2, 3, 0, 1], :, :].clone()
                # mask_list[1] = torch.sigmoid(mask.clone())
                # merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                flow_d, mask, conf = self.block2(
                    warp(img0, flow[:, :2]),
                    warp(img1, flow[:, 2:4]),
                    warp(f0, flow[:, :2]),
                    warp(f1, flow[:, 2:4]),
                    timestep,
                    mask,
                    flow,
                    scale=scale[2],
                    cnd = sharpness
                )
                flow = flow + flow_d

                flow_list[2] = flow[:, [2, 3, 0, 1], :, :].clone()
                # mask_list[2] = torch.sigmoid(mask.clone())
                # merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_d, mask, conf = self.block3(
                    warp(img0, flow[:, :2]),
                    warp(img1, flow[:, 2:4]),
                    warp(f0, flow[:, :2]),
                    warp(f1, flow[:, 2:4]),
                    timestep,
                    mask,
                    flow,
                    scale=scale[3],
                    cnd = sharpness
                )
                flow = flow + flow_d

                flow_list[3] = flow[:, [2, 3, 0, 1], :, :]
                # mask_list[3] = torch.sigmoid(mask)
                # merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

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