class Model:
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

        '''
        def warp(tenInput, tenFlow):
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
        '''

        def warp(tenInput, tenFlow):
            input_device = tenInput.device
            input_dtype = tenInput.dtype
            if 'mps' in str(input_device):
                tenInput = tenInput.detach().to(device=torch.device('cpu'), dtype=torch.float32)
                tenFlow = tenFlow.detach().to(device=torch.device('cpu'), dtype=torch.float32)

            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device, dtype=tenInput.dtype)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            result = torch.nn.functional.grid_sample(
                input=tenInput, 
                grid=g, 
                mode='bilinear', 
                padding_mode='border', 
                align_corners=True
                )

            return result.detach().to(device=input_device, dtype=input_dtype)

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, feat=False):
                x0 = self.cnn0(x)
                x = self.relu(x0)
                x1 = self.cnn1(x)
                x = self.relu(x1)
                x2 = self.cnn2(x)
                x = self.relu(x2)
                x3 = self.cnn3(x)
                if feat:
                    return [x0, x1, x2, x3]
                return x3

        class Encode01(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3+8, 32, 3, 2, 1)
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)

            def forward(self, x, feat=False):
                x0 = self.cnn0(x)
                x = self.relu(x0)
                x1 = self.cnn1(x)
                x = self.relu(x1)
                x2 = self.cnn2(x)
                x = self.relu(x2)
                x3 = self.cnn3(x)
                if feat:
                    return [x0, x1, x2, x3]
                return x3

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)
                
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class LastConvBlock(Module):
            def __init__(self, in_planes, c):
                super().__init__()

                self.lastconv2 = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                    conv(c//2, c//2, 3, 1, 1),
                    torch.nn.ConvTranspose2d(c//2, c//4, 4, 2, 1),
                    conv(c//4, 5, 3, 1, 1),
                    # torch.nn.PixelShuffle(2)
                )

                # self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True) 
                # self.downconv = conv(c+in_planes, (c+in_planes)*2, 3, 2, 1)
                # self.multires_deep01 = MultiresblockRev((c+in_planes)*2, (c+in_planes)*2)
                # self.upsample_deep01 = torch.nn.ConvTranspose2d((c+in_planes)*2, c, 4, 2, 1)
                 #self.relu = torch.nn.LeakyReLU(0.2, True)

                # self.multires01 = MultiresblockRevNoact(c, c)
                # self.upsample01 = torch.nn.ConvTranspose2d(c*2, c, 4, 2, 1)
                # self.multires02 = MultiresblockRevNoact(c, c)
                # self.upsample02 = torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1)
                # self.multires03 = MultiresblockRevNoact(c//2+5, c//2+5, shortcut_bias=False)
                # self.conv_final = Conv2d(c//2+5, 5, kernel_size = (3,3))

            def forward(self, feat, x):
                tmp_rife = self.lastconv2(feat)
                return tmp_rife

                tmp_deep = self.downconv(torch.cat([feat, x], dim=1))
                tmp_deep = self.multires_deep01(tmp_deep)
                tmp_deep = self.upsample_deep01(tmp_deep)

                tmp_refine = self.multires01(feat)
                tmp_refine = self.upsample01(torch.cat((tmp_refine, tmp_deep), dim=1))
                tmp_refine = self.multires02(tmp_refine)
                tmp_refine = self.upsample02(tmp_refine)

                tmp_refine = self.multires03(torch.cat((tmp_rife, tmp_refine), dim=1))

                out = self.conv_final(tmp_refine)

                # feat = self.relu(tmp_deep * self.beta + feat)
                # out = self.lastconv2(feat)

                '''
                
                tmp_refine = self.upsample01(torch.cat((tmp_refine, tmp_deep), dim=1))
                tmp_refine = self.multires02(tmp_refine)
                tmp_refine = self.upsample02(tmp_refine)

                tmp_refine = self.multires03(torch.cat((tmp_rife, tmp_refine), dim=1))
                out = self.conv_final(tmp_refine)
                '''

                return out


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
                self.lastconv2 = LastConvBlock(in_planes, c)
                self.encode = Head()

            def forward(self, img0, img1, timestep, mask, flow, scale=1):
                img0 = torch.nn.functional.interpolate(img0, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                img1 = torch.nn.functional.interpolate(img1, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                if flow is None:
                    timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                    x = torch.cat((img0, img1, f0, f1, timestep), 1)
                else:
                    mask = torch.nn.functional.interpolate(mask, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    img0 = warp(img0, flow[:, :2])
                    img1 = warp(img1, flow[:, 2:4])
                    f0 = warp(f0, flow[:, :2])
                    f1 = warp(f1, flow[:, 2:4])
                    timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                    x = torch.cat((img0, img1, f0, f1, timestep, mask, flow), 1)

                feat = self.conv0(x)
                feat = self.convblock(feat)

                x = torch.nn.functional.interpolate(x, scale_factor= 1 / 4, mode="bilinear", align_corners=False) * 1 / 4
                x[:, :-4] *= 1 / 4
                out = self.lastconv2(feat, x)

                out = torch.nn.functional.interpolate(out, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = out[:, :4] * scale
                mask = out[:, 4:5]

                return flow, mask

        class FlownetCas(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(7+16, c=192)
                self.block1 = Flownet(8+4+16, c=128)
                self.block2 = Flownet(8+4+16, c=96)
                self.block3 = Flownet(8+4+16, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[8, 4, 2, 1], iterations=1):
                img0 = img0
                img1 = img1

                flow_list = [None] * 4
                mask_list = [None] * 4
                merged = [None] * 4
                flow, mask = self.block0(img0, img1, timestep, None, None, scale=scale[0])

                flow_list[0] = flow
                mask_list[0] = torch.sigmoid(mask)
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                for iteration in range(iterations):
                    flow_d, mask = self.block1(
                        img0, 
                        img1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[1]
                    )
                    flow += flow_d

                flow_list[1] = flow
                mask_list[1] = torch.sigmoid(mask)
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                for iteration in range(iterations):
                    flow_d, mask = self.block2(
                        img0, 
                        img1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[2]
                    )
                    flow += flow_d

                flow_list[2] = flow
                mask_list[2] = torch.sigmoid(mask)
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                for iteration in range(iterations):
                    flow_d, mask = self.block3(
                        img0, 
                        img1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[3]
                    )
                    flow += flow_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                return flow_list, mask_list, merged

        class FlownetMem(Module):
            def __init__(self):
                super().__init__()
                self.block0 = Flownet(7+16, c=192)
                self.block1 = Flownet(8+4+16, c=128)
                self.block2 = Flownet(8+4+16, c=96)
                self.block3 = Flownet(8+4+16, c=64)
                self.encode = Head()

            def forward(self, img0, img1, timestep=0.5, scale=[8, 4, 2, 1], iterations=1):
                img0 = img0
                img1 = img1
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                merged = [None] * 4
                flow, mask = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0])

                for iteration in range(iterations):
                    flow_d, mask = self.block1(
                        warp(img0, flow[:, :2]), 
                        warp(img1, flow[:, 2:4]),
                        warp(f0, flow[:, :2]),
                        warp(f1, flow[:, 2:4]),
                        timestep,
                        mask,
                        flow, 
                        scale=scale[1]
                        )
                    flow += flow_d
                    del flow_d

                for iteration in range(iterations):
                    flow_d, mask = self.block2(
                        warp(img0, flow[:, :2]), 
                        warp(img1, flow[:, 2:4]),
                        warp(f0, flow[:, :2]),
                        warp(f1, flow[:, 2:4]),
                        timestep,
                        mask,
                        flow, 
                        scale=scale[2]
                        )
                    flow += flow_d
                    del flow_d

                for iteration in range(iterations):
                    flow_d, mask = self.block3(
                        warp(img0, flow[:, :2]), 
                        warp(img1, flow[:, 2:4]),
                        warp(f0, flow[:, :2]),
                        warp(f1, flow[:, 2:4]),
                        timestep,
                        mask,
                        flow, 
                        scale=scale[3]
                        )
                    flow += flow_d
                    del flow_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                # merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                return flow_list, mask_list, merged


        self.model = FlownetMem
        self.training_model = FlownetCas

    @staticmethod
    def get_info():
        info = {
            'name': 'Flownet4_v003',
            'file': 'flownet4_v003.py',
            'ratio_support': True
        }
        return info

    @staticmethod
    def get_name():
        return 'TWML_Flownet_v003'

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
        # import platform
        # if platform.system() == 'Darwin':
        #     return self.training_model
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