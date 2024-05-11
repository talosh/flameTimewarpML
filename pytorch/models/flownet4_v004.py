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
                    bias=bias
                    )
                torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                self.conv1.weight.data *= 1e-2
                if self.conv1.bias is not None:
                    torch.nn.init.constant_(self.conv1.bias, 0)
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
                    bias=True
                    )
                self.act = torch.nn.LeakyReLU(0.2, True)
                torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                self.conv1.weight.data *= 1e-2
                if self.conv1.bias is not None:
                    torch.nn.init.constant_(self.conv1.bias, 0)
                # torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
                # torch.nn.init.dirac_(self.conv1.weight)

            def forward(self,x):
                x = self.conv1(x)
                x = self.act(x)
                return x

        class Conv2d_SiLU(Module):
            def __init__(self, num_in_filters, num_out_filters, kernel_size, stride = (1,1)):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(
                    in_channels=num_in_filters,
                    out_channels=num_out_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding = 'same',
                    padding_mode = 'reflect',
                    bias=True
                    )
                self.act = torch.nn.SiLU(inplace=True)
                # torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='selu')
                # torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
                # torch.nn.init.dirac_(self.conv1.weight)

            def forward(self,x):
                x = self.conv1(x)
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
    
        class Multiresblock(Module):
            def __init__(self, num_in_channels, num_filters, alpha=1, shortcut_bias = True):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_3x3 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                # filt_cnt_7x7 = int(self.W*0.5)
                filt_cnt_7x7 = num_filters - (filt_cnt_3x3 + filt_cnt_5x5)

                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
                
                self.shortcut = Conv2d(
                    num_in_channels,
                    num_out_filters, 
                    kernel_size = (1,1),
                    bias = shortcut_bias
                    )

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

        class MultiresblockRev(Module):
            def __init__(self, num_in_channels, num_filters, alpha=1, shortcut_bias = True):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_7x7 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                # filt_cnt_7x7 = int(self.W*0.5)
                filt_cnt_3x3 = num_filters - (filt_cnt_7x7 + filt_cnt_5x5)

                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
                
                self.shortcut = Conv2d(
                    num_in_channels,
                    num_out_filters, 
                    kernel_size = (1,1),
                    bias = shortcut_bias
                    )

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

        class MultiresblockRevNoact(Module):
            def __init__(self, num_in_channels, num_filters, alpha=1, shortcut_bias = True):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_7x7 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                # filt_cnt_7x7 = int(self.W*0.5)
                filt_cnt_3x3 = num_filters - (filt_cnt_7x7 + filt_cnt_5x5)

                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
                
                self.shortcut = Conv2d(
                    num_in_channels,
                    num_out_filters, 
                    kernel_size = (1,1),
                    bias = shortcut_bias
                    )

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
                # x = self.act(x)
            
                return x

        class ResConvBlock(Module):
            def __init__(self, num_in_channels, num_filters, alpha=1, shortcut_bias = True):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_3x3 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                # filt_cnt_7x7 = int(self.W*0.5)
                filt_cnt_7x7 = num_filters - (filt_cnt_3x3 + filt_cnt_5x5)
                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
                
                self.shortcut = Conv2d(
                    num_in_channels,
                    num_out_filters, 
                    kernel_size = (3,3),
                    )

                self.conv_3x3 = Conv2d_ReLU(num_in_channels, filt_cnt_3x3, kernel_size = (3,3))

                self.conv_5x5 = Conv2d_ReLU(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3))
                
                self.conv_7x7 = Conv2d_ReLU(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3))

                self.act = torch.nn.LeakyReLU(0.2, True)

                self.joinconv = Conv2d(
                    num_filters*2,
                    num_out_filters, 
                    kernel_size = (3,3),
                    )

            def forward(self,x):

                shrtct = self.shortcut(x)
                
                a = self.conv_3x3(x)
                b = self.conv_5x5(a)
                c = self.conv_7x7(b)

                x = torch.cat([a,b,c],axis=1)

                x = self.joinconv(torch.cat([x, shrtct],axis=1))
            
                return x

        class ResConvBlockRev(Module):
            def __init__(self, num_in_channels, num_filters, alpha=1, shortcut_bias = True):
            
                super().__init__()
                self.alpha = alpha
                self.W = num_filters * alpha
                
                filt_cnt_7x7 = int(self.W*0.167)
                filt_cnt_5x5 = int(self.W*0.333)
                # filt_cnt_7x7 = int(self.W*0.5)
                filt_cnt_3x3 = num_filters - (filt_cnt_7x7 + filt_cnt_5x5)
                num_out_filters = filt_cnt_3x3 + filt_cnt_5x5 + filt_cnt_7x7
                
                self.shortcut = Conv2d(
                    num_in_channels,
                    num_out_filters, 
                    kernel_size = (3,3),
                    )

                self.conv_3x3 = Conv2d_ReLU(num_in_channels, filt_cnt_3x3, kernel_size = (3,3))

                self.conv_5x5 = Conv2d_ReLU(filt_cnt_3x3, filt_cnt_5x5, kernel_size = (3,3))
                
                self.conv_7x7 = Conv2d_ReLU(filt_cnt_5x5, filt_cnt_7x7, kernel_size = (3,3))

                self.act = torch.nn.LeakyReLU(0.2, True)

                self.joinconv = Conv2d(
                    num_filters*2,
                    num_out_filters, 
                    kernel_size = (3,3),
                    )

            def forward(self,x):

                shrtct = self.shortcut(x)
                
                a = self.conv_3x3(x)
                b = self.conv_5x5(a)
                c = self.conv_7x7(b)

                x = torch.cat([a,b,c],axis=1)

                x = self.joinconv(torch.cat([x, shrtct],axis=1))
            
                return x


        class ResConv(Module):
            def __init__(self, c, dilation=1):
                super().__init__()
                self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=True)
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True) # torch.nn.SELU(inplace = True)
                
            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)

        class MultiResConv(Module):
            def __init__(self, c):
                super().__init__()
                self.conv = torch.nn.Sequential(
                    ResConvBlock(c, c),
                    ResConvBlockRev(c, c)
                )
                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
                self.relu = torch.nn.LeakyReLU(0.2, True)
                
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

                self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True) 
                self.downconv = conv(c+in_planes, (c+in_planes)*2, 3, 2, 1)
                self.multires_deep01 = MultiresblockRev((c+in_planes)*2, (c+in_planes)*2)
                self.upsample_deep01 = torch.nn.ConvTranspose2d((c+in_planes)*2, c, 4, 2, 1)
                self.relu = torch.nn.LeakyReLU(0.2, True)

                '''
                self.multires01 = MultiresblockRevNoact(c, c)
                self.upsample01 = torch.nn.ConvTranspose2d(c*2, c, 4, 2, 1)
                self.multires02 = MultiresblockRevNoact(c, c)
                self.upsample02 = torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1)
                self.multires03 = MultiresblockRevNoact(c//2+5, c//2+5, shortcut_bias=False)
                self.conv_final = Conv2d(c//2+5, 5, kernel_size = (3,3))
                '''

            def forward(self, feat, x):
                tmp_deep = self.downconv(torch.cat([feat, x], dim=1))
                tmp_deep = self.multires_deep01(tmp_deep)
                tmp_deep = self.upsample_deep01(tmp_deep)
                feat = self.relu(tmp_deep * self.beta + feat)
                out = self.lastconv2(feat)

                '''
                tmp_refine = self.multires01(torch.cat([feat, x], dim=1))
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
                self.encode01 = torch.nn.Sequential(
                    torch.nn.Conv2d(3, 32, 3, 2, 1),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.Conv2d(32, 32, 3, 1, 1),
                    torch.nn.LeakyReLU(0.2, True),
                    torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                )
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
                    MultiResConv(c),
                )

                self.lastconv2 = LastConvBlock(in_planes, c)

            def forward(self, img0, img1, timestep, mask, flow, scale=1):

                img0 = torch.nn.functional.interpolate(img0, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                img1 = torch.nn.functional.interpolate(img1, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                f0 = self.encode01(img0)
                f1 = self.encode01(img1)

                if flow is None:
                    x = torch.cat((img0, img1, f0, f1, timestep), 1)
                else:
                    mask = torch.nn.functional.interpolate(mask, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    img0 = warp(img0, flow[:, :2])
                    img1 = warp(img1, flow[:, 2:4])
                    f0 = warp(f0, flow[:, :2])
                    f1 = warp(f1, flow[:, 2:4])
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

            def forward(self, img0, gt, img1, f0_0, f1_0, timestep=0.5, scale=[8, 4, 2, 1]):
                print ('hello')


                img0 = img0
                img1 = img1

                flow_list = [None] * 4
                mask_list = [None] * 4
                merged = [None] * 4
                flow, mask = self.block0(img0, img1, timestep, None, None, scale=scale[0])
                flow_list[0] = flow
                mask_list[0] = torch.sigmoid(mask)
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])
        
                flow_d, mask = self.block1(img0, img1, timestep, mask, flow, scale=scale[1])
                flow = flow + flow_d

                flow_list[1] = flow
                mask_list[1] = torch.sigmoid(mask)
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                flow_d, mask = self.block2(img0, img1, timestep, mask, flow, scale=scale[2])
                flow = flow + flow_d

                flow_list[2] = flow
                mask_list[2] = torch.sigmoid(mask)
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                flow_d, mask = self.block3(img0, img1, timestep, mask, flow, scale=scale[3])
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
            'name': 'Flownet4_v004',
            'file': 'flownet4_v004.py',
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