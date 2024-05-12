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

        def warp(tenInput, tenFlow):
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device)
            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

        class Conv2d(Module):
            def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1), bias=True):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=num_in_filters,
                    out_channels=num_out_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding = 'same',
                    padding_mode = 'reflect',
                    bias=False
                    )
                torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                # torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
                # torch.nn.init.dirac_(self.conv1.weight)

            def forward(self,x):
                x = self.conv1(x)
                return x

        class Conv2d_ReLU(Module):
            def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1)):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=num_in_filters,
                    out_channels=num_out_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding = 'same',
                    padding_mode = 'reflect',
                    bias=False
                    )
                self.act = torch.nn.LeakyReLU(0.2, True)
                torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                # torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
                # torch.nn.init.dirac_(self.conv1.weight)

            def forward(self,x):
                x = self.conv1(x)
                x = self.act(x)
                return x

        class Multiresblock(Module):
            '''
            MultiRes Block
            
            Arguments:
                num_in_channels {int} -- Number of channels coming into mutlires block
                num_filters {int} -- Number of filters in a corrsponding UNet stage
                alpha {float} -- alpha hyperparameter (default: 1.67)
            
            '''

            def __init__(self, num_in_channels, num_filters, alpha=1.69):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_3x3 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                filt_cnt_7x7 = int(self.W*0.5)
                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
                
                self.shortcut = Conv2d(num_in_channels ,num_out_filters , kernel_size = (1,1))

                self.conv_3x3 = Conv2d_ReLU(num_in_channels, filt_cnt_3x3, kernel_size = (3,3))

                self.conv_5x5 = Conv2d_ReLU(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3))
                
                self.conv_7x7 = Conv2d_ReLU(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3))

                self.act = torch.nn.LeakyReLU(0.2, True)

            def forward(self,x):

                shrtct = self.shortcut(x)
                
                a = self.conv_3x3(x)
                b = self.conv_5x5(a)
                c = self.conv_7x7(b)

                x = torch.cat([a,b,c],axis=1)

                x = x + shrtct
                x = self.act(x)
            
                return x


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

        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)
                
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

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
                    torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1),
                    conv(c//2, c//2, 3, 1, 1),
                    torch.nn.ConvTranspose2d(c//2, c//4, 4, 2, 1),
                    conv(c//4, 5, 3, 1, 1),
                )

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                x = torch.cat((img0, img1, f0, f1, timestep), 1)
                x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                if flow is not None:
                    mask = torch.nn.functional.interpolate(mask, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, mask, flow), 1)
                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
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
                f0 = self.encode(img0)
                f1 = self.encode(img1)

                flow_list = [None] * 4
                mask_list = [None] * 4
                merged = [None] * 4
                flow, mask = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0])
                flow_list[0] = flow
                mask_list[0] = torch.sigmoid(mask)
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

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
                    flow = flow + flow_d

                flow_list[1] = flow
                mask_list[1] = torch.sigmoid(mask)
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

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
                    flow = flow + flow_d

                flow_list[2] = flow
                mask_list[2] = torch.sigmoid(mask)
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

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
                    flow = flow + flow_d

                flow_list[3] = flow
                mask_list[3] = torch.sigmoid(mask)
                merged[3] = warp(img0, flow[:, :2]) * mask_list[3] + warp(img1, flow[:, 2:4]) * (1 - mask_list[3])

                return flow_list, mask_list, merged

        self.model = FlownetCas
        self.training_model = FlownetCas

    @staticmethod
    def get_info():
        info = {
            'name': 'Flownet4_v002',
            'file': 'flownet4_v002.py',
            'ratio_support': True
        }
        return info

    @staticmethod
    def get_name():
        return 'TWML_Flownet_v001'

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