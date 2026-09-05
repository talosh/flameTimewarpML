# Beatles block in spatial domain + additional freq pass at 1/2 res

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import random
import math

class Model:

    info = {
        'name': 'Stabnet4_v009',
        'file': 'stabnet4_v009.py',
        'ratio_support': True
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        backwarp_tenGrid = {}

        def overlay(base, blend):
            return torch.where(
                base < 0.5,
                2 * base * blend,
                1 - 2 * (1 - base) * (1 - blend)
            )

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

            gkernel = gauss_kernel(img.shape[1])
            gkernel = gkernel.to(device=img.device, dtype=img.dtype)
            hp = img - conv_gauss(img, gkernel)

            # hp is zero-mean — normalize by per-image max abs deviation, then shift to [0, 1]
            # scale = hp.abs().flatten(1).max(dim=1).values + 1e-6   # (B,)
            # scale = scale.view(-1, 1, 1, 1)                         # broadcast over C, H, W
            # hp = hp / scale * 0.5 + 0.5
            
            hp = hp + 0.5

            return hp

        def warp(tenInput, tenFlow):
            backwarp_tenGrid = {}
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device = tenInput.device, dtype = tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        def normflow(tenFlow):
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0) ], 1)
            return tenFlow

        class Head(Module):
            def __init__(self, c=24):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3, c, 9, 2, 4, bias=False)
                self.norm0 = torch.nn.GroupNorm(num_groups=12, num_channels=c)

                self.cnn1 = torch.nn.Conv2d(c, c, 3, 1, 1, bias=False)
                self.norm1 = torch.nn.GroupNorm(num_groups=12, num_channels=c)
                self.cnn2 = torch.nn.Conv2d(c, c, 3, 1, 1, bias=False)
                self.norm2 = torch.nn.GroupNorm(num_groups=12, num_channels=c)

                self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.cnn3 = torch.nn.Conv2d(c, 8, 3, 1, 1, bias=False)
                self.norm3 = torch.nn.GroupNorm(num_groups=8, num_channels=8)

                self.act = torch.nn.SiLU()
                
                self.maxdepth = 2

            def forward(self, x):
                # x = x + 2.0 * (hpass(x)) - 1.0
                n, c, h, w = x.shape
                ph = self.maxdepth - (h % self.maxdepth)
                pw = self.maxdepth - (w % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)

                x = self.act(self.norm0(self.cnn0(x)))
                x = self.act(self.norm1(self.cnn1(x)))
                x = self.act(self.norm2(self.cnn2(x)))

                x = self.norm3(self.up(self.cnn3(x)))

                return x[:, :, :h, :w]

        class SwiGLU(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.inp = torch.nn.Conv2d(c, 8 * c, 1)
                self.out = torch.nn.Conv2d(4 * c, c, 1)

            def forward(self, x):
                gate, value = self.inp(x).chunk(2, dim=1)
                x = torch.nn.functional.silu(gate) * value
                x = self.out(x)
                return x

        class ResConv(nn.Module):
            def __init__(self, c, dropout=0.1):
                super().__init__()
                self.cnn = torch.nn.Conv2d(c, c, 3, 1, 1, bias=False)                
                self.beta = torch.nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
                self.norm = nn.GroupNorm(math.gcd(24, c), c)
                self.act = torch.nn.SiLU()
                # self.act = SwiGLU(c)
                # self.freq = FourierUnit(c, c)

            def forward(self, x):
                h = self.act(self.norm(x))
                h = self.cnn(h)
                # f = self.freq(h)
                # self.cnn(torch.cat([f, h], 1)) * self.beta
                #  x = self.act(self.norm(self.conv(x)) * self.beta + x)
                return x + h

        class FourierUnit(nn.Module):
            def __init__(self, c, g):
                super().__init__()
                self.channels = c
                self.groups = g
                self.norm = torch.nn.GroupNorm(math.gcd(24, 2*c), 2*c)
                self.fp = torch.nn.Conv2d(2*c, 2*c, 3, padding = 1, groups = 2*c, bias=True)
                self.fc = torch.nn.Conv2d(2*c, 2*c * g, 1, groups = g, bias=True)
                self.gw = torch.nn.Sequential(
                    torch.nn.Conv2d(2*c, g, 1),
                    torch.nn.Softmax(dim=1)
                )
                self.act = torch.nn.SiLU()

            def forward(self, x):
                b, c, h, w = x.shape
                sp = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
                sp = torch.view_as_real(sp)
                sp = sp.permute(0, 1, 4, 2, 3).reshape(b, 2*c, h, -1)
                sp = self.norm(sp)
                sp = sp + self.fp(sp)
                wt = self.gw(sp)
                cand = self.fc(sp)
                fwidth = cand.shape[-1]
                cand = cand.reshape(b, self.groups, 2 * c, h, fwidth)
                sp = (cand * wt.unsqueeze(2)).sum(dim=1)
                sp = self.act(sp)
                sp = (sp.reshape(b, c, 2, h, fwidth)).permute(0, 1, 3, 4, 2).contiguous()
                sp = torch.view_as_complex(sp)
                return torch.fft.irfft2(sp, s=(h,w), dim=(-2, -1), norm='ortho')

        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.cnn0 = torch.nn.Sequential(
                    torch.nn.Conv2d(in_planes, c // 4, 9, 2, 4, bias=True),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(c // 4, c // 2, 3, 2, 1, bias=True),
                    torch.nn.SiLU(),
                )


                self.cnn1 = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=False),
                    torch.nn.Conv2d(in_planes, c // 2, 9, 2, 4, bias=True),
                    torch.nn.SiLU(),
                    torch.nn.Conv2d(c // 2, c, 3, 2, 1, bias=True),
                    torch.nn.SiLU(),
                )

                self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                # self.up = torch.nn.PixelShuffle(4)

                # self.up_d = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
                # self.cnn3 = torch.nn.Conv2d(2 * c, c // 2, 9, 1, 4, bias=True)
                # self.cnn4 = torch.nn.Conv2d(c // 2, 4, 9, 1, 4, bias=True)

                self.mix = torch.nn.Sequential(
                    torch.nn.Conv2d(c + c // 2, c // 2, 3, 1, 1, bias=True),
                    torch.nn.SiLU()
                )

                self.head = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                    torch.nn.Conv2d(c // 2, 4, 3, 1, 1, bias=True),
                    # torch.nn.Tanh()
                )

                # torch.nn.init.normal_(self.head[-1].weight, mean=0.0, std=4)
                # torch.nn.init.zeros_(self.head[-1].bias)

                self.convblock = torch.nn.Sequential(
                    ResConv(c // 2),
                    ResConv(c // 2),
                    ResConv(c // 2),
                    ResConv(c // 2),
                )

                self.convblock_d = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )

                '''
                self.convblock_f = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                '''
                '''
                self.convblock_l = torch.nn.Sequential(
                    ResConv(c // 2),
                    ResConv(c // 2),
                    ResConv(c // 2),
                    ResConv(c // 2),
                )
                '''

                '''
                self.convblock_p = torch.nn.Sequential(
                    ResConv(c // 2),
                    ResConv(c // 2),
                )
                '''

                self.maxdepth = 4

            def forward(self, img0, img1, flow=None, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round((h * (1 / scale)) / 16) * 16, round((w * (1 / scale)) / 16 ) * 16

                # ph = self.maxdepth - (sh % self.maxdepth)
                # pw = self.maxdepth - (sw % self.maxdepth)
                # padding = (0, pw, 0, ph)

                # imgs = torch.cat((img0, img1), 1)
                # imgs = normalize(imgs, 0, 1) * 2 - 1
                # x = torch.cat((imgs, f0, f1), 1)

                x = torch.cat((img0, img1), 1)

                if flow is not None:
                    # flow = flow / scale
                    x = torch.cat((x, flow), 1)

                # x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                # x = torch.nn.functional.pad(x, padding)
                
                feat = self.cnn0(x)
                feat_d = self.cnn1(x)

                # print (f'feat: {feat.shape}')
                # print (f'featd: {feat_d.shape}')

                # feat = self.convblock_p(feat)
                # feat_d = torch.cat([feat, self.freq(feat)], 1)

                '''
                feat_f = torch.fft.rfft2(feat, dim=(-2, -1))
                feat_f = torch.cat([feat_f.real, feat_f.imag], dim=1)
                feat_f = self.convblock_f(feat_f)
                feat_f = torch.fft.irfft2(
                    torch.complex(*feat_f.chunk(2, dim=1)),
                    s=feat.shape[-2:],
                    dim=(-2, -1),
                    norm='ortho'
                )
                '''
                # feat_f = torch.functional.interpolate(feat_f, size=feat.shape[-2:], mode='bilinear', align_corners=False)

                
                feat = self.convblock(feat)
                feat_d = self.convblock_d(feat_d)

                # feat = self.up(feat)
                # feat_f = self.up(feat_f)
                # feat = self.up(feat)
                feat_d = self.up(feat_d)

                # print (f'featd_up: {feat_d.shape}')

                # feat = self.act(self.cnn3(torch.cat([feat, feat_d, feat_f], 1)))
                # feat = self.cnn4(self.up(feat))
                # feat = self.mix(torch.cat([feat, feat_d, feat_f], 1))
                feat = self.mix(torch.cat([feat, feat_d], 1))
                # feat = self.convblock_l(feat)
                # feat = self.head(feat)
                feat = self.head(feat)

                # print (f'feat_head: {feat.shape}')

                # import sys
                # sys.exit()

                '''
                feat = feat * feat.new_tensor([
                    (feat.shape[3] - 1.0) / 2.0,
                    (feat.shape[2] - 1.0) / 2.0,
                ]).repeat(2).view(1, 4, 1, 1)
                '''

                # feat = torch.nn.functional.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
                flow =  feat * scale

                return flow

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(6, c=192)
                # self.block1 = Flownet(48, c=144)
                # self.block2 = Flownet(48, c=108)
                # self.block3 = Flownet(48, c=96)
                # self.encode = Head()

            def flow_to_uv(self, flow):
                b, _, h, w = flow.shape

                x = torch.linspace(-1.0, 1.0, w, device=flow.device, dtype=flow.dtype)
                y = torch.linspace(-1.0, 1.0, h, device=flow.device, dtype=flow.dtype)

                grid_x = x.view(1, 1, 1, w).expand(b, 1, h, w)
                grid_y = y.view(1, 1, h, 1).expand(b, 1, h, w)
                base_grid = torch.cat([grid_x, grid_y], dim=1)

                flow_x = flow[:, 0:1] * 2.0 / (w - 1)
                flow_y = flow[:, 1:2] * 2.0 / (h - 1)
                flow_norm = torch.cat([flow_x, flow_y], dim=1)

                grid = base_grid + flow_norm
                grid_inv = base_grid - flow_norm

                uv_x = (grid[:, 0:1] + 1.0) / 2.0
                uv_y = 1.0 - (grid[:, 1:2] + 1.0) / 2.0
                uv = torch.cat([uv_x, uv_y], dim=1)

                return uv

            def forward(self, img0, img1, scale=[4, 3, 2, 1]):
                '''
                def halving_steps(start, n_steps = 4):
                    vals = [max(start // (2 ** i), 1) for i in range(n_steps - 1)]
                    vals.append(1)
                    return vals

                def skewed_randint(low=1, high=16, alpha=1.0, beta=2.0):
                    x = random.betavariate(alpha, beta)
                    return low + min(int(x * (high - low + 1)), high - low)

                scale = halving_steps(skewed_randint(), 4)

                # scale = halving_steps(random.randint(1, 16), 4)
                # scale = [round(v) for v in torch.linspace(random.randint(4, 16), 1, steps=4).tolist()]
                
                f0 = self.encode(img0)
                f1 = self.encode(img1)
                '''

                # img0 = torch.arcsinh(img0 * 2 - 1)
                # img1 = torch.arcsinh(img1 * 2 - 1)

                scale = [1]

                flow_list = []
                flow0 = self.block0(img0, img1, scale = scale[0])

                # flow0 = self.block0(img0, img1, f0, f1, scale = scale[0])

                flow_list.append(flow0.clone())

                result = {
                    'flow_list': flow_list,
                    'scale': scale
                }

                return result

                '''
                flow_fwd, flow_rev = torch.split(flow0, 2, dim=1)
                img1_fwd = warp(img1, flow_fwd)
                img1_rev = warp(img1_fwd, flow_rev)
                f1_fwd = warp(f1, flow_fwd)
                f1_rev = warp(f1_fwd, flow_rev)
                
                flow1 = flow0 + self.block1(
                    torch.cat([img0, img1], 1),
                    torch.cat([img1_fwd, img1_rev], 1),
                    torch.cat([f0, f1], 1), 
                    torch.cat([f1_fwd, f1_rev], 1), 
                    torch.cat([normflow(flow_fwd), normflow(flow_rev)], 1),
                    scale = scale[1]
                    )

                flow_list.append(flow1.clone())
                flow_fwd, flow_rev = torch.split(flow1, 2, dim=1)
                img1_fwd = warp(img1, flow_fwd)
                img1_rev = warp(img1_fwd, flow_rev)
                f1_fwd = warp(f1, flow_fwd)
                f1_rev = warp(f1_fwd, flow_rev)

                flow2 = flow1 + self.block2(
                    torch.cat([img0, img1], 1),
                    torch.cat([img1_fwd, img1_rev], 1),
                    torch.cat([f0, f1], 1),
                    torch.cat([f1_fwd, f1_rev], 1),
                    torch.cat([normflow(flow_fwd), normflow(flow_rev)], 1),
                    scale = scale[2]
                    )

                flow_list.append(flow2.clone())
                flow_fwd, flow_rev = torch.split(flow2, 2, dim=1)
                img1_fwd = warp(img1, flow_fwd)
                img1_rev = warp(img1_fwd, flow_rev)
                f1_fwd = warp(f1, flow_fwd)
                f1_rev = warp(f1_fwd, flow_rev)

                flow3 = flow2 + self.block3(
                    torch.cat([img0, img1], 1),
                    torch.cat([img1_fwd, img1_rev], 1),
                    torch.cat([f0, f1], 1),
                    torch.cat([f1_fwd, f1_rev], 1),
                    torch.cat([normflow(flow_fwd), normflow(flow_rev)], 1),
                    scale = scale[3]
                    )

                flow_list.append(flow3.clone())

                result = {
                    'flow_list': flow_list,
                    'scale': scale
                }

                return result
                '''

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
                flownet.load_state_dict(convert(torch.load(path)), False)
            else:
                flownet.load_state_dict(convert(torch.load(path, map_location ='cpu')), False)