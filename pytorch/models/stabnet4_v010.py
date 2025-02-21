# Beatles block in spatial domain + additional freq pass at 1/2 res

class Model:

    info = {
        'name': 'Stabnet4_v010',
        'file': 'stabnet4_v010.py',
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
                torch.nn.PReLU(out_planes, 0.2)
            )

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

        def to_freq(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.float()
            x = torch.fft.rfft2(x, dim=(-2, -1))  # Compute real-input FFT2
            x = torch.cat([x.real.unsqueeze(2), x.imag.unsqueeze(2)], dim=2).view(n, c * 2, h, w // 2 + 1)
            x = x.to(dtype=src_dtype)
            return x

        def to_spat(x):
            n, c, h, w_half = x.shape  # w_half corresponds to (w//2 + 1)
            src_dtype = x.dtype
            x = x.float()
            x = x.view(n, c // 2, 2, h, w_half)  # Restore real & imaginary parts
            x = torch.complex(x[:, :, 0, :, :], x[:, :, 1, :, :])  # Create complex tensor
            x = torch.fft.irfft2(x, s=(h, (w_half - 1) * 2), dim=(-2, -1))  # Inverse FFT2
            x = x.to(dtype=src_dtype)
            return x

        class Head(Module):
            def __init__(self, c=32):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(3, c, 3, 2, 1),
                    torch.nn.PReLU(c, 0.2),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    torch.nn.ConvTranspose2d(c, 8, 4, 2, 1)
                )
                self.maxdepth = 2

            def forward(self, x):
                hp = hpass(x)
                x = normalize(x, 0, 1) * 2 - 1
                n, c, h, w = x.shape
                ph = self.maxdepth - (h % self.maxdepth)
                pw = self.maxdepth - (w % self.maxdepth)
                padding = (0, pw, 0, ph)
                x = torch.nn.functional.pad(x, padding)
                x = self.encode(x)[:, :, :h, :w]
                x = torch.cat((x, hp), 1)
                return x

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.PReLU(c, 0.2)
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class UpMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(c, 0.2)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x_deep) * self.beta + x)

        class Mix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv0 = torch.nn.ConvTranspose2d(cd, c, 4, 2, 1)
                self.conv1 = torch.nn.Conv2d(c, c, 3, 1, 1)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(c, 0.2)

            def forward(self, x, x_deep):
                return self.relu(self.conv0(x_deep) * self.beta + self.conv1(x) * self.gamma)

        class DownMix(Module):
            def __init__(self, c, cd):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, cd, 3, 2, 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, cd, 1, 1)), requires_grad=True)
                self.relu = torch.nn.PReLU(cd, 0.2)

            def forward(self, x, x_deep):
                return self.relu(self.conv(x) * self.beta + x_deep)

        class FlownetDeepDualHead(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                cd = 1 * round(1.618 * c) + 2 - (1 * round(1.618 * c) % 2)
                self.conv0 = conv(in_planes, c//2, 3, 2, 1)
                self.conv1 = conv(c//2, c, 3, 2, 1)
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
                self.convblock1f = torch.nn.Sequential(
                    ResConv(c//2),
                    ResConv(c//2),
                    ResConv(c//2),
                    ResConv(c//2),
                )
                self.convblock2f = torch.nn.Sequential(
                    ResConv(c//2),
                    ResConv(c//2),
                    ResConv(c//2),
                )
                self.convblock3f = torch.nn.Sequential(
                    ResConv(c//2),
                    ResConv(c//2),
                )
                self.convblock_last = torch.nn.Sequential(
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                    ResConv(c),
                )
                self.convblock_last_deep = torch.nn.Sequential(
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
                    ResConv(cd),
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
                
                self.mix1 = UpMix(c, cd)
                self.mix1f = DownMix(c//2, c)
                self.mix2 = UpMix(c, cd)
                self.mix2f = DownMix(c//2, c)
                self.mix3 = Mix(c, cd)
                self.mix3f = DownMix(c//2, c)
                self.revmix1 = DownMix(c, cd)
                self.revmix1f = UpMix(c//2, c)
                self.revmix2 = DownMix(c, cd)
                self.revmix2f = UpMix(c//2, c)
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*2, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                    # torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                    # torch.nn.PReLU(c//2, 0.2),
                    # torch.nn.Conv2d(c//2, c//2, 3, 1, 1),
                    # torch.nn.PReLU(c//2, 0.2),
                    # torch.nn.ConvTranspose2d(c//2, 2, 4, 2, 1)
                )
                self.maxdepth = 8

            def forward(self, img0, img1, f0, f1, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)

                imgs = torch.cat((img0, img1), 1)
                imgs = normalize(imgs, 0, 1) * 2 - 1
                x = torch.cat((imgs, f0, f1), 1)
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
                x = torch.nn.functional.pad(x, padding)
                
                '''
                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1).to(device=img0.device, dtype=img0.dtype)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.nn.functional.pad(tenGrid, padding, mode='replicate')
                x = torch.cat((x, tenGrid), 1)
                '''

                feat = self.conv0(x)
                featF = self.convblock1f(feat)
                feat = self.conv1(feat)
                feat_deep = self.conv2(feat)

                feat = self.convblock1(feat)
                feat_deep = self.convblock_deep1(feat_deep)
                
                feat = self.mix1f(featF, feat)
                feat_tmp = self.mix1(feat, feat_deep)
                feat_deep = self.revmix1(feat, feat_deep)
                featF = self.revmix1f(featF, feat_tmp)

                featF = self.convblock2f(featF)
                feat = self.convblock2(feat_tmp)
                feat_deep = self.convblock_deep2(feat_deep)

                feat = self.mix2f(featF, feat)
                feat_tmp = self.mix2(feat, feat_deep)
                feat_deep = self.revmix2(feat, feat_deep)
                featF = self.revmix2f(featF, feat_tmp)

                featF = self.convblock3f(featF)
                feat = self.convblock3(feat_tmp)
                feat_deep = self.convblock_deep3(feat_deep)

                feat = self.mix3f(featF, feat)
                feat = self.mix3(feat, feat_deep)

                feat = self.convblock_last(feat)
                feat = self.lastconv(feat)
                feat = torch.nn.functional.interpolate(feat[:, :, :sh, :sw], size=(h, w), mode="bilinear", align_corners=False)
                flow = feat * scale
                return flow

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = FlownetDeepDualHead(24, c=64) # images + feat + timestep + lingrid
                # self.block1 = FlownetDeepDualHead(6+3+20+1+1+2+1+4, 12+20+8+1, c=128) # FlownetDeepDualHead(9+30+1+1+4+1+2, 22+30+1, c=128) # images + feat + timestep + lingrid + mask + conf + flow
                # self.block2 = FlownetDeepDualHead(6+3+20+1+1+2+1+4, 12+20+8+1, c=96) # FlownetLT(6+2+1+1+1, c=48) # None # FlownetDeepDualHead(9+30+1+1+4+1+2, 22+30+1, c=112) # images + feat + timestep + lingrid + mask + conf + flow
                # self.block3 = FlownetLT(11, c=48)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[16, 8, 4, 1], iterations=4, gt=None):
                img0 = compress(img0 * 2 - 1)
                img1 = compress(img1 * 2 - 1)

                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow = self.block0(img0, img1, f0, f1, scale=1)

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