# Orig v001 changed to v002 main flow and signatures
# Back from SiLU to LeakyReLU to test data flow
# Warps moved to flownet forward
# Different Tail from flownet 2lh (ConvTr 6x6, conv 1x1, ConvTr 4x4, conv 1x1)

class Model:

    info = {
        'name': 'Stabnet4_v001fa_v04_001',
        'file': 'stabnet4_v001fa_v04_001.py',
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
                    padding_mode = 'reflect',
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
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bicubic', padding_mode='border', align_corners=True)

        def normalize(tensor, min_val, max_val):
            src_dtype = tensor.dtype
            tensor = tensor.float()
            t_min = tensor.min()
            t_max = tensor.max()

            if t_min == t_max:
                return torch.full_like(tensor, (min_val + max_val) / 2.0)
            
            tensor = ((tensor - t_min) / (t_max - t_min)) * (max_val - min_val) + min_val
            tensor = tensor.to(dtype = src_dtype)
            return tensor

        def hpass(img):
            src_dtype = img.dtype
            img = img.float()
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
            hp = img - conv_gauss(img, gkernel) + 0.5
            hp = torch.clamp(hp, 0.48, 0.52)
            hp = normalize(hp, 0, 1)
            hp = torch.max(hp, dim=1, keepdim=True).values
            hp = hp.to(dtype = src_dtype)
            return hp

        def compress(x):
            src_dtype = x.dtype
            x = x.float()
            scale = torch.tanh(torch.tensor(1.0))
            x = torch.where(
                (x >= -1) & (x <= 1), scale * x,
                torch.tanh(x)
            )
            x = (x + 1) / 2
            x = x.to(dtype = src_dtype)
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
            x = x.float()
            x = torch.fft.fft2(x, dim=(-2, -1))
            x = torch.fft.fftshift(x, dim=(-2, -1))
            x = torch.cat([x.real.unsqueeze(2), x.imag.unsqueeze(2)], dim=2).view(n, c * 2, h, w)
            x = x.to(dtype = src_dtype)
            return x

        def to_spat(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.float()
            x = x.view(n, c//2, 2, h, w)
            x = torch.complex(
                x[:, :, 0, :, :],
                x[:, :, 1, :, :]
            )
            x = torch.fft.ifftshift(x, dim=(-2, -1))
            x = torch.fft.ifft2(x, dim=(-2, -1)).real
            x = x.to(dtype=src_dtype)
            return x
        
        class HeadF(Module):
            def __init__(self):
                super().__init__()
                self.cnn0f = torch.nn.Conv2d(6, 48, 3, 2, 1)
                self.cnn1f = torch.nn.Conv2d(48, 48, 3, 1, 1)
                self.cnn2f = torch.nn.Conv2d(48, 48, 3, 1, 1)
                self.cnn3f = torch.nn.ConvTranspose2d(48, 10, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x):
                x = x * 2 - 1
                x = x - x.mean()
                xf = self.cnn0f(to_freq(x))
                xf = self.relu(xf)
                xf = self.cnn1f(xf)
                xf = self.relu(xf)
                xf = self.cnn2f(xf)
                xf = self.relu(xf)
                xf = self.cnn3f(xf)
                return xf

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(4, 32, 3, 2, 1),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.ConvTranspose2d(32, 10, 4, 2, 1)
                )
                self.maxdepth = 2

            def forward(self, x):
                hp = hpass(x)
                x = torch.cat((x, hp), 1)

                n, c, h, w = x.shape
                ph = self.maxdepth - (h % self.maxdepth)
                pw = self.maxdepth - (w % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)

                return self.encode(x)[:, :, :h, :w]

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class UpMixToFreq(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(cd, c//2, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(to_freq(self.conv(x_deep)) * self.beta + x)

        class UpMixToSpat(Module):
            def __init__(self, c, cd):
                super().__init__()
                # self.conv0 = torch.nn.Conv2d(c//2, c, 3, 1, 1)
                self.conv0 = torch.nn.ConvTranspose2d(cd, c//2, 4, 2, 1)
                self.conv1 = torch.nn.ConvTranspose2d(c//2, c//2, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c//2, 1, 1)), requires_grad=True)
                self.gamma = torch.nn.Parameter(torch.ones((1, c//2, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv0(x_deep) * self.beta + self.conv1(to_spat(x)) * self.gamma)

        class DownMixToSpat(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.Conv2d(c//2, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, x_deep):
                return self.relu(self.conv(to_spat(x)) * self.beta + x_deep)

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
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, conf, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                
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
            def __init__(self, in_planes, c=48):
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
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.maxdepth = 4

            def forward(self, img0, img1, timestep, mask, conf, flow, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))
                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)

                timestep = (img0[:, :1].clone() * 0 + 1) * timestep

                img0 = torch.cat((img0, hpass(img0)), 1)
                img1 = torch.cat((img1, hpass(img1)), 1)
                
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                x = torch.cat((warped_img0, warped_img1, timestep, mask, conf), 1)
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bilinear", align_corners=False)
                # flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                # x = torch.cat((x, flow), 1)

                x = torch.nn.functional.pad(x, padding, mode='constant')

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)[:, :, :sh, :sw]
                tmp = torch.nn.functional.interpolate(tmp[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

        class FlownetDeepDualHead(Module):
            def __init__(self, in_planes, in_planes_fx, c=64):
                super().__init__()
                cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)
                self.conv0 = conv(in_planes, c//2, 3, 2, 1)
                self.conv0f = conv(in_planes_fx, c, 3, 2, 1)
                self.conv1 = conv(c//2, c, 3, 2, 1)
                self.conv1f = conv(c, c, 3, 2, 1)
                self.conv2 = conv(c, cd, 3, 2, 1)
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
                    ResConv(c//2),
                    ResConv(c//2),
                    ResConv(c//2),
                    ResConv(c//2),
                )
                self.convblock_deep1 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep2 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                )
                self.convblock_deep3 = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                )
                
                self.mix1 = UpMixToFreq(c, cd)
                self.mix2 = UpMixToFreq(c, cd)
                self.mix3 = UpMixToSpat(c, cd)
                self.revmix1 = DownMixToSpat(c, cd)
                self.revmix2 = DownMixToSpat(c, cd)
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c//2, 4*2, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, flow=None, scale=1, encode_xf=None):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)

                if flow is None:
                    imgs = torch.cat((img0, img1), 1)
                    imgs = normalize(imgs, 0, 1) * 2 - 1
                    x = torch.cat((imgs, f0, f1), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
                    x = torch.nn.functional.pad(x, padding)
                    
                    img0_scaled = torch.nn.functional.interpolate(img0, size=(sh, sw), mode="bicubic", align_corners=False)
                    img1_scaled = torch.nn.functional.interpolate(img1, size=(sh, sw), mode="bicubic", align_corners=False)
                    img0_scaled = torch.nn.functional.pad(img0_scaled, padding)
                    img1_scaled = torch.nn.functional.pad(img1_scaled, padding)

                    f0xf = encode_xf(img0_scaled)
                    f1xf = encode_xf(img1_scaled)
                    imgs_scaled = torch.cat((img0_scaled, img1_scaled), 1)
                    imgs_scaled = normalize(imgs_scaled, 0, 1) * 2 - 1
                    xf = torch.cat((to_freq(imgs_scaled), f0xf, f1xf), 1)

                else:
                    merged = warp(img0, flow[:, :2]) * torch.sigmoid(mask) + warp(img1, flow[:, 2:4]) * (1 - torch.sigmoid(mask))
                    imgs = torch.cat((img0, img1, merged), 1)
                    imgs = normalize(imgs, 0, 1) * 2 - 1
                    x = torch.cat((imgs, f0, f1, mask, conf), 1)
                    x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, size=(sh, sw), mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)
                    x = torch.nn.functional.pad(x, padding)

                    img0_scaled = torch.nn.functional.interpolate(img0, size=(sh, sw), mode="bicubic", align_corners=False)
                    img1_scaled = torch.nn.functional.interpolate(img1, size=(sh, sw), mode="bicubic", align_corners=False)
                    img0_scaled = torch.nn.functional.pad(img0_scaled, padding)
                    img1_scaled = torch.nn.functional.pad(img1_scaled, padding)
                    flow = torch.nn.functional.pad(flow, padding)

                    f0xf = encode_xf(img0_scaled)
                    f1xf = encode_xf(img1_scaled)
                    imgs_scaled = torch.cat((img0_scaled, img1_scaled), 1)
                    imgs_scaled = normalize(imgs_scaled, 0, 1) * 2 - 1
                    xf = torch.cat((to_freq(imgs_scaled), f0xf, f1xf, to_freq(flow)), 1)

                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1).to(device=img0.device, dtype=img0.dtype)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.nn.functional.pad(tenGrid, padding, mode='replicate')
                x = torch.cat((x, tenGrid), 1)

                feat = self.conv0f(xf)
                feat_deep = self.conv0(x)
                feat_deep = self.conv1(feat_deep)
                feat_deep = self.conv2(feat_deep)

                # potential attention or insertion here
                feat = self.conv1f(feat)
                feat = self.convblock1(feat)
                feat_deep = self.convblock_deep1(feat_deep)

                feat_tmp = self.mix1(feat, feat_deep)
                feat_deep = self.revmix1(feat, feat_deep)

                feat = self.convblock2(feat_tmp)
                feat_deep = self.convblock_deep2(feat_deep)

                feat_tmp = self.mix2(feat, feat_deep)
                feat_deep = self.revmix2(feat, feat_deep)

                feat = self.convblock3(feat_tmp)
                feat_deep = self.convblock_deep3(feat_deep)

                feat = self.mix3(feat, feat_deep)
                feat = self.convblock_last(feat)
                feat = self.lastconv(feat)
                feat = torch.nn.functional.interpolate(feat[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = feat * scale
                return flow

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeepDualHead(6+20+2, 6+20+2+4, c=96) # images + feat + timestep + lingrid
                # self.block1 = FlownetDeepDualHead(6+3+20+1+1+2+1+4, 12+20+8+1, c=128) # FlownetDeepDualHead(9+30+1+1+4+1+2, 22+30+1, c=128) # images + feat + timestep + lingrid + mask + conf + flow
                # self.block2 = FlownetDeepDualHead(6+3+20+1+1+2+1+4, 12+20+8+1, c=96) # FlownetLT(6+2+1+1+1, c=48) # None # FlownetDeepDualHead(9+30+1+1+4+1+2, 22+30+1, c=112) # images + feat + timestep + lingrid + mask + conf + flow
                # self.block3 = FlownetLT(11, c=48)
                self.encode = Head()
                self.encode_xf = HeadF()

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], iterations=4, gt=None):
                img0 = compress(img0 * 2 - 1)
                img1 = compress(img1 * 2 - 1)

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow = self.block0(img0, img1, f0, f1, scale=1, encode_xf=self.encode_xf)

                result = {
                    'flow_list': [flow]
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
                flownet.load_state_dict(convert(torch.load(path)), False)
            else:
                flownet.load_state_dict(convert(torch.load(path, map_location ='cpu')), False)