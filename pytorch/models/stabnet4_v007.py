# vanilla RIFE block + PReLU in FreqDomain

class Model:

    info = {
        'name': 'Stabnet4_v007',
        'file': 'stabnet4_v007.py',
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

        '''
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
        '''

        '''
        def to_freq(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.float()
            x = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')  # Compute real-input FFT2
            magnitude = torch.abs(x)  # Compute magnitude
            phase = torch.angle(x)  # Compute phase
            x = torch.cat([magnitude.unsqueeze(2), phase.unsqueeze(2)], dim=2).view(n, c * 2, h, w // 2 + 1)  # Fix shape issue
            x = x.to(dtype=src_dtype)
            return x

        def to_spat(x):
            n, c, h, w_half = x.shape  # w_half corresponds to (w//2 + 1)
            src_dtype = x.dtype
            x = x.float()
            x = x.view(n, c // 2, 2, h, w_half)  # Restore real & imaginary parts
            magnitude = x[:, :, 0, :, :]
            phase = x[:, :, 1, :, :]
            phase = torch.clamp(phase, -torch.pi, torch.pi)
            x = torch.polar(magnitude, phase)  # Convert magnitude and phase back to complex
            x = torch.fft.irfft2(x, s=(h, w_half * 2 - 1), dim=(-2, -1), norm='ortho')  # Fix width reconstruction
            x = x.to(dtype=src_dtype)
            return x
        '''

        def to_freq(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.float()
            x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')  # Compute full FFT2
            real_part = x.real  # Extract real part
            imag_part = x.imag  # Extract imaginary part
            x = torch.cat([real_part.unsqueeze(2), imag_part.unsqueeze(2)], dim=2).view(n, c * 2, h, w)  # Keep full width
            x = x.to(dtype=src_dtype)
            return x

        def to_spat(x):
            n, c, h, w = x.shape
            src_dtype = x.dtype
            x = x.float()
            x = x.view(n, c // 2, 2, h, w)  # Restore real & imaginary parts
            real_part = x[:, :, 0, :, :]
            imag_part = x[:, :, 1, :, :]
            x = torch.complex(real_part, imag_part)  # Convert back to complex
            x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho').real  # Compute inverse FFT2, take only real part
            x = x.to(dtype=src_dtype)
            return x

        class SpectralConv2d(Module):
            def __init__(self, in_channels, out_channels, modes1, modes2):
                super(SpectralConv2d, self).__init__()

                """
                2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
                """

                self.in_channels = in_channels
                self.out_channels = out_channels
                self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
                self.modes2 = modes2

                self.scale = (1 / (in_channels * out_channels))
                self.weights1 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
                self.weights2 = torch.nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

            # Complex multiplication
            def compl_mul2d(self, input, weights):
                # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
                return torch.einsum("bixy,ioxy->boxy", input, weights)

            def forward(self, x):
                batchsize = x.shape[0]
                #Compute Fourier coeffcients up to factor of e^(- something constant)
                x_ft = torch.fft.rfft2(x)

                # Multiply relevant Fourier modes
                out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
                out_ft[:, :, :self.modes1, :self.modes2] = \
                    self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
                out_ft[:, :, -self.modes1:, :self.modes2] = \
                    self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

                #Return to physical space
                x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
                return x

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.encode = torch.nn.Sequential(
                    torch.nn.Conv2d(4, 32, 3, 2, 1),
                    torch.nn.PReLU(32, 0.2),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    torch.nn.PReLU(32, 0.2),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    torch.nn.PReLU(32, 0.2),
                    torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
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
                self.conv = SpectralConv2d(c, c, 11, 11) # torch.nn.Conv2d(c, c, 3, 1, 1, dilation = dilation, groups = 1, padding_mode = 'zeros', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.gamma = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
                self.relu = torch.nn.LeakyReLU(0.2, True)
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x * self.gamma)

        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 3, 2, 1),
                    conv(c//2, c, 3, 2, 1),
                    )
                self.convblock = torch.nn.Sequential(
                    ResConv(2*c),
                    ResConv(2*c),
                    ResConv(2*c),
                    ResConv(2*c),
                    ResConv(2*c),
                    ResConv(2*c),
                    ResConv(2*c),
                    ResConv(2*c),
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*2, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )
                self.maxdepth = 4

            def forward(self, img0, img1, f0, f1, scale=1):
                n, c, h, w = img0.shape
                sh, sw = round(h * (1 / scale)), round(w * (1 / scale))

                ph = self.maxdepth - (sh % self.maxdepth)
                pw = self.maxdepth - (sw % self.maxdepth)
                padding = (0, pw, 0, ph)

                img0 = normalize(img0, 0, 1) * 2 - 1
                img1 = normalize(img1, 0, 1) * 2 - 1
                img0 = img0 - img0.mean()
                img1 = img1 - img1.mean()
                imgs = torch.cat((img0, img1), 1)
                # imgs = normalize(imgs, 0, 1) * 2 - 1
                x = torch.cat((imgs, f0, f1), 1)
                # x = imgs
                x = torch.nn.functional.interpolate(x, size=(sh, sw), mode="bicubic", align_corners=False)
                x = torch.nn.functional.pad(x, padding)

                tenHorizontal = torch.linspace(-1.0, 1.0, sw).view(1, 1, 1, sw).expand(n, -1, sh, -1).to(device=img0.device, dtype=img0.dtype)
                tenVertical = torch.linspace(-1.0, 1.0, sh).view(1, 1, sh, 1).expand(n, -1, -1, sw).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.nn.functional.pad(tenGrid, padding, mode='replicate')
                x = torch.cat((x, tenGrid), 1)

                feat = self.conv0(x)
                # feat = to_freq(feat)
                feat = self.convblock(feat)
                # feat = to_spat(feat)

                '''
                _, _, shf, swf = feat.shape
                tenHorizontal = torch.linspace(-1.0, 1.0, swf).view(1, 1, 1, swf).expand(n, -1, shf, -1).to(device=img0.device, dtype=img0.dtype)
                tenVertical = torch.linspace(-1.0, 1.0, shf).view(1, 1, shf, 1).expand(n, -1, -1, swf).to(device=img0.device, dtype=img0.dtype)
                tenGrid = torch.cat((tenHorizontal, tenVertical), 1).to(device=img0.device, dtype=img0.dtype)
                feat = torch.cat((feat, tenGrid), 1)
                '''

                feat = self.lastconv(feat)[:, :, :sh, :sw]

                # feat = to_spat(feat[:, :, :sh, :sw])
                feat = torch.nn.functional.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)
                flow = feat * scale
                # flow = torch.cat([ flow[:, 0:1, :, :] * ((flow.shape[3] - 1.0) / 2.0), flow[:, 1:2, :, :] * ((flow.shape[2] - 1.0) / 2.0) ], 1)
                return flow

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(6+16+2, c=64) # images + feat + timestep + lingrid
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