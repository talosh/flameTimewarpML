# Orig v001 changed to v002 main flow and signatures
# Back from SiLU to LeakyReLU to test data flow
# Warps moved to flownet forward
# Different Tail from flownet 2lh (ConvTr 6x6, conv 1x1, ConvTr 4x4, conv 1x1)

class Model:

    info = {
        'name': 'Flownet4_v001f',
        'file': 'flownet4_v001f.py',
        'ratio_support': True
    }

    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module
        backwarp_tenGrid = {}

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_planes, 
                    out_planes, 
                    kernel_size=kernel_size, 
                    stride=stride,
                    padding=padding, 
                    dilation=dilation,
                    padding_mode = 'zeros',
                    bias=True
                ),
                torch.nn.LeakyReLU(0.2, True)
                # torch.nn.SELU(inplace = True)
            )

        def warp(tenInput, tenFlow):
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        def hpass(img):
            def gauss_kernel(size=5, channels=3):
                kernel = torch.tensor([[1., 4., 6., 4., 1],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]])
                kernel /= 256.
                kernel = kernel.repeat(channels, 1, 1, 1)
                return kernel
            
            def conv_gauss(img, kernel):
                img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
                out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
                return out

            def normalize(tensor, min_val, max_val):
                return (tensor - min_val) / (max_val - min_val)

            gkernel = gauss_kernel()
            gkernel = gkernel.to(device=img.device, dtype=img.dtype)
            hp = img - conv_gauss(img, gkernel) + 0.5
            hp = torch.clamp(hp, 0.48, 0.52)
            hp = normalize(hp, hp.min(), hp.max())
            hp = torch.max(hp, dim=1, keepdim=True).values
            return hp

        def normalize(x):
            x = x * 2 - 1
            scale = torch.tanh(torch.tensor(1.0))
            x = torch.where(
                (x >= -1) & (x <= 1), scale * x,
                torch.tanh(x)
            )
            x = (x + 1) / 2
            return x

        def compress(x):
            scale = torch.tanh(torch.tensor(1.0))
            x = torch.where(
                (x >= -1) & (x <= 1), scale * x,
                torch.tanh(x)
            )
            x = (x + 1) / 2
            return x

        def blur(img):  
            def gauss_kernel(size=5, channels=3):
                kernel = torch.tensor([[1., 4., 6., 4., 1],
                                    [4., 16., 24., 16., 4.],
                                    [6., 24., 36., 24., 6.],
                                    [4., 16., 24., 16., 4.],
                                    [1., 4., 6., 4., 1.]])
                kernel /= 256.
                kernel = kernel.repeat(channels, 1, 1, 1)
                return kernel
            
            def conv_gauss(img, kernel):
                img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
                out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
                return out

            gkernel = gauss_kernel()
            gkernel = gkernel.to(device=img.device, dtype=img.dtype)
            return conv_gauss(img, gkernel)

        def to_freq(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = torch.fft.fft2(x.float(), dim=(-2, -1))
            x = torch.fft.fftshift(x, dim=(-2, -1))
            x = torch.cat([x.real.unsqueeze(2), x.imag.unsqueeze(2)], dim=2).view(n, c * 2, h, w)
            x = x.to(dtype = src_dtype)
            return x

        def to_spat(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.view(n, c//2, 2, h, w).float()
            x = torch.complex(
                x[:, :, 0, :, :],
                x[:, :, 1, :, :]
            )
            x = torch.fft.ifftshift(x, dim=(-2, -1))
            x = torch.fft.ifft2(x, dim=(-2, -1)).real
            x = x.to(dtype=src_dtype)
            return x

        def to_freq_aa(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = torch.fft.fft2(x.float(), dim=(-2, -1))
            x = torch.fft.fftshift(x, dim=(-2, -1))
            # x = torch.cat([x.real.unsqueeze(2), x.imag.unsqueeze(2)], dim=2).view(n, c * 2, h, w)
            x = torch.cat([x.abs().unsqueeze(2), x.angle().unsqueeze(2)], dim=2).view(n, c * 2, h, w)
            x = x.to(dtype = src_dtype)
            return x

        def to_spat_aa(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.view(n, c//2, 2, h, w).float()
            x = torch.complex(
                x[:, :, 0, :, :] * torch.cos(x[:, :, 1, :, :]),
                x[:, :, 0, :, :] * torch.sin(x[:, :, 1, :, :])
            )
            x = torch.fft.ifftshift(x, dim=(-2, -1))
            x = torch.fft.ifft2(x, dim=(-2, -1)).real
            x = x.to(dtype=src_dtype)
            return x
        
        class Head(Module):
            def __init__(self):
                super().__init__()
                self.cnn0 = torch.nn.Conv2d(4, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 10, 4, 2, 1)
                self.relu = torch.nn.Mish(True)

            def forward(self, x):
                hp = hpass(x)
                hp = hp.to(dtype = x.dtype)
                x = torch.cat((x, hp), 1)
            
                x = self.cnn0(x * 2 - 1)
                x = self.relu(x)
                x = self.cnn1(x)
                x = self.relu(x)
                x = self.cnn2(x)
                x = self.relu(x)

                x = self.cnn3(x)
                return x

        class HeadFreq(Module):
            def __init__(self):
                super().__init__()
                self.cnn0f = torch.nn.Conv2d(8, 48, 3, 2, 1)
                self.cnn1f = torch.nn.Conv2d(48, 48, 3, 1, 1)
                self.cnn2f = torch.nn.Conv2d(48, 48, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(48, 8, 4, 2, 1)
                self.relu = torch.nn.Mish(True)

            def forward(self, x):
                hp = hpass(x)
                hp = hp.to(dtype = x.dtype)
                x = torch.cat((x, hp), 1)

                xf = self.cnn0f(to_freq(x * 2 - 1))
                xf = self.relu(xf)
                xf = self.cnn1f(xf)
                xf = self.relu(xf)
                xf = self.cnn2f(xf)
                xf = self.relu(xf)
                xf = self.cnn3(xf)
                return xf

        class HeadMixed(Module):
            def __init__(self):
                super().__init__()
                self.cnn0 = torch.nn.Conv2d(4, 32, 3, 2, 1)
                self.cnn0f = torch.nn.Conv2d(8, 48, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn1f = torch.nn.Conv2d(48, 48, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2f = torch.nn.Conv2d(48, 48, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(56, 10, 4, 2, 1)
                self.relu = torch.nn.Mish(True)

            def forward(self, x):
                hp = hpass(x)
                hp = hp.to(dtype = x.dtype)
                x = torch.cat((x, hp), 1)

                xf = self.cnn0f(to_freq(x * 2 - 1))
                xf = self.relu(xf)
                xf = self.cnn1f(xf)
                xf = self.relu(xf)
                xf = self.cnn2f(xf)
                xf = self.relu(xf)
                xf = to_spat(xf)

                x = self.cnn0(x * 2 - 1)
                x = self.relu(x)
                x = self.cnn1(x)
                x = self.relu(x)
                x = self.cnn2(x)
                x = self.relu(x)
                
                x = torch.cat((x, xf), 1)
                x = self.cnn3(x)
                return x

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class ResConvMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x_deep) * self.beta + x)

        class ResConvRevMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x) * self.beta + x_deep)

        class LastConv(Module):
            def __init__(self, c, c_out):
                super().__init__()
                self.up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                self.up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
                self.conv1 = conv(c, c//2, 3, 1, 1)
                self.conv2 = conv(c//2, c//4, 3, 1, 1)
                self.lastconv = torch.nn.Conv2d(c//2, c_out, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                n, c, h, w = x.shape
                d = self.up4(x[:, :c // 4, :, :])
                x = self.conv1(self.up2(x))
                x = self.conv2(self.up2(x))
                x = torch.cat((d, x), 1)
                return self.lastconv(x)

        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 5, 2, 2),
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
                    torch.nn.ConvTranspose2d(c, c, 6, 2, 2),
                    torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
                    torch.nn.ConvTranspose2d(c, c, 4, 2, 1),
                    torch.nn.Conv2d(c, 6, kernel_size=1, stride=1, padding=0, bias=True),
                )
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, conf, scale=1, gt = None, fgt = None):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                
                if gt is not None:
                    if flow is None:
                        x = torch.cat((img0, gt, img1, f0, fgt, f1, timestep), 1)
                        x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                    else:
                        warped_img0 = warp(img0, flow[:, :2])
                        warped_img1 = warp(img1, flow[:, 2:4])
                        warped_f0 = warp(f0, flow[:, :2])
                        warped_f1 = warp(f1, flow[:, 2:4])
                        x = torch.cat((warped_img0, gt, warped_img1, warped_f0, fgt, warped_f1, timestep, mask, conf), 1)
                        x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                        flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                        x = torch.cat((x, flow), 1)
                else:
                    if flow is None:
                        x = torch.cat((img0, img1, f0, f1, timestep), 1)
                        x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                    else:
                        warped_img0 = warp(img0, flow[:, :2])
                        warped_img1 = warp(img1, flow[:, 2:4])
                        warped_f0 = warp(f0, flow[:, :2])
                        warped_f1 = warp(f1, flow[:, 2:4])
                        x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask, conf), 1)
                        x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                        flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                        x = torch.cat((x, flow), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetLT(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c, 3, 2, 1),
                    conv(c, c, 3, 2, 1),
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
                    torch.nn.ConvTranspose2d(c, c, 6, 2, 2),
                    torch.nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0, bias=True),
                    torch.nn.ConvTranspose2d(c, c, 4, 2, 1),
                    torch.nn.Conv2d(c, 6, kernel_size=1, stride=1, padding=0, bias=True),
                )
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                x = torch.cat((warped_img0, warped_img1, timestep, mask), 1)
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                x = torch.cat((x, flow), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding, mode='constant')

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetDeepSingleHead(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = int(1.618 * c)
                self.conv0 = conv(in_planes, c//2, 5, 2, 2)
                self.conv0ft = conv(in_planes, c, 5, 2, 2)
                self.conv1 = conv(c//2, c, 3, 2, 1)
                self.conv1ft = conv(c, 2*c, 3, 2, 1)
                self.conv2 = conv(2*c, 2*cd, 3, 2, 1)
                self.convblock1 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock2 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock3 = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock_last = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock_deep1 = torch.nn.Sequential(
                    ResConv(2*cd),
                    ResConv(2*cd),
                    ResConv(2*cd),
                    ResConv(2*cd),
                )
                self.convblock_deep2 = torch.nn.Sequential(
                    ResConv(2*cd),
                    ResConv(2*cd),
                    ResConv(2*cd),
                )
                self.convblock_deep3 = torch.nn.Sequential(
                    ResConv(2*cd),
                    ResConv(2*cd),
                )
                
                self.mix1 = ResConvMix(c, cd)
                self.mix2 = ResConvMix(c, cd)
                self.mix3 = ResConvMix(c, cd)
                self.revmix1 = ResConvRevMix(2*c, 2*cd)
                self.revmix2 = ResConvRevMix(2*c, 2*cd)
                # self.lastconv = LastConv(c, 6)
                self.lastconv = torch.nn.Sequential(
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    conv(c, c//2, 3, 1, 1),
                    conv(c//2, c//2, 3, 1, 1),
                    torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    torch.nn.Conv2d(c//2, 6, kernel_size=3, stride=1, padding=1)
                    # torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    # torch.nn.PixelShuffle(2)
                )
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, conf, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                if flow is None:
                    x = torch.cat((img0, img1, f0, f1), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img1, flow[:, 2:4])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, mask, conf), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)

                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw)
                tenGrid = torch.cat((
                    tenHorizontal * ((sw - 1.0) / 2.0), 
                    tenVertical * ((sh - 1.0) / 2.0)
                    ), 1).to(device=img0.device, dtype=img0.dtype)
                timestep = (tenGrid[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((x, timestep, tenGrid), 1)

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)

                feat = self.conv0(x)
                feat_ft = self.conv0ft(to_freq(x))
                # potential attention or insertion here
                feat_ft = self.conv1ft(feat_ft)
                feat_deep = self.conv2(feat_ft)

                feat = self.conv1(feat)

                feat = self.convblock1(feat)
                feat_deep = self.convblock_deep1(feat_deep)

                feat_tmp = self.mix1(feat, to_spat(feat_deep))
                feat_deep = self.revmix1(to_freq(feat), feat_deep)

                feat = self.convblock2(feat_tmp)
                feat_deep = self.convblock_deep2(feat_deep)

                feat_tmp = self.mix2(feat, to_spat(feat_deep))
                feat_deep = self.revmix2(to_freq(feat), feat_deep)

                feat = self.convblock3(feat_tmp)
                feat_deep = self.convblock_deep3(feat_deep)

                feat = self.mix3(feat, to_spat(feat_deep))

                feat = self.convblock_last(feat)

                feat_tmp = self.lastconv(feat)

                feat_tmp = torch.nn.functional.interpolate(feat_tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = feat_tmp[:, :4] * scale
                mask = feat_tmp[:, 4:5]
                conf = feat_tmp[:, 5:6]
                return flow, mask, conf

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeepSingleHead(6+20+1+2, c=192) # images + feat + timestep + lineargrid
                self.block0ref = None # FlownetDeepSingleHead(6+18+1+1+1+4+2, c=192) # images + feat + timestep + mask + conf + flow + lineargrid
                self.block1 = None # Flownet(6+18+1+1+1+4, c=144) # images + feat + timestep + mask + conf + flow
                self.block2 = None # Flownet(6+18+1+1+1+4, c=96)
                self.block3 = None # Flownet(6+18+1+1+1+4, c=64)
                self.block_distill = Flownet(6+20+1+1+1+4+3+10, c=96) # images + feat + timestep + mask + conf + flow + gt + fgt
                self.encode = HeadMixed()

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], iterations=1, gt=None):
                # src_dtype = img0.dtype
                # img0 = img0.float()
                # img1 = img1.float()

                img0 = normalize(img0)
                img1 = normalize(img1)
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                conf_list = [None] * 4
                merged = [None] * 4

                scale[0] = 1

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, None, scale=scale[0])

                # flow = flow.to(dtype = src_dtype)
                # mask = mask.to(dtype = src_dtype)
                # conf = conf.to(dtype = src_dtype)
                # img0 = img0.to(dtype = src_dtype)
                # img1 = img1.to(dtype = src_dtype)

                flow_list[3] = flow
                conf_list[3] = compress(conf) # torch.sigmoid(conf)
                mask_list[3] = compress(mask) # torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                flow_dist = None
                merged_dist = None
                gt_distill = None

                if gt is not None:
                    gt_distill = normalize(gt)
                    fgt = self.encode(gt_distill)
                    flow_dist, mask_dist, conf_dist = self.block_distill(
                        img0, 
                        img1,
                        f0,
                        f1,
                        timestep,
                        mask,
                        flow,
                        conf,
                        scale=scale[3],
                        gt = gt_distill,
                        fgt = fgt
                    )
                    flow_dist = flow + flow_dist
                    mask_dist = compress(mask_dist)
                    merged_dist = warp(img0, flow_dist[:, :2]) * mask_dist + warp(img1, flow_dist[:, 2:4]) * (1 - mask_dist)

                result = {
                    'flow_list': flow_list,
                    'mask_list': mask_list,
                    'conf_list': conf_list,
                    'merged': merged,
                    'flow_distill': flow_dist,
                    'merged_distill': merged_dist,
                    'gt_distill': gt_distill
                }

                return result

                flow, mask, conf = self.block0ref(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask_init,
                    flow_init,
                    conf_init,
                    scale=scale[0]
                )

                flow_list[0] = flow.clone()
                conf_list[0] = torch.sigmoid(conf.clone())
                mask_list[0] = torch.sigmoid(mask.clone())
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                flow_d, mask_d, conf_d = self.block1(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow,
                    conf,
                    scale=scale[1]
                )
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[1] = flow.clone()
                conf_list[1] = torch.sigmoid(conf.clone())
                mask_list[1] = torch.sigmoid(mask.clone())
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                flow_d, mask_d, conf_d = self.block2(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow,
                    conf,
                    scale=scale[2]
                )
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[2] = flow.clone()
                conf_list[2] = torch.sigmoid(conf.clone())
                mask_list[2] = torch.sigmoid(mask.clone())
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_d, mask_d, conf_d = self.block3(
                    img0, 
                    img1,
                    f0,
                    f1,
                    timestep,
                    mask,
                    flow,
                    conf,
                    scale=scale[3]
                )
                flow = flow + flow_d
                mask = mask + mask_d
                conf = conf + conf_d

                flow_list[3] = flow
                conf_list[3] = torch.sigmoid(conf)
                mask_list[3] = torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                return flow_list, mask_list, conf_list, merged

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