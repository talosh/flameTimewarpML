# Orig v001 changed to v002 main flow and signatures
# SiLU in Encoder
# Warps moved to flownet forward
# Replaced ResBlocks with CBAM blocks
# Resblock with Spatial awareness only
# Spatial kernel set to 5

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

        class Head(Module):
            def __init__(self):
                super(Head, self).__init__()
                self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1, padding_mode = 'reflect')
                self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode = 'reflect')
                self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode = 'reflect')
                self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
                self.relu = torch.nn.SiLU(True)

                torch.nn.init.kaiming_normal_(self.cnn0.weight, mode='fan_in', nonlinearity='relu')
                self.cnn0.weight.data *= 1e-2
                if self.cnn0.bias is not None:
                    torch.nn.init.constant_(self.cnn0.bias, 0)
                torch.nn.init.kaiming_normal_(self.cnn1.weight, mode='fan_in', nonlinearity='relu')
                self.cnn1.weight.data *= 1e-2
                if self.cnn1.bias is not None:
                    torch.nn.init.constant_(self.cnn1.bias, 0)
                torch.nn.init.kaiming_normal_(self.cnn2.weight, mode='fan_in', nonlinearity='relu')
                self.cnn2.weight.data *= 1e-2
                if self.cnn2.bias is not None:
                    torch.nn.init.constant_(self.cnn2.bias, 0)
                torch.nn.init.kaiming_normal_(self.cnn3.weight, mode='fan_in', nonlinearity='relu')
                self.cnn3.weight.data *= 1e-2
                if self.cnn3.bias is not None:
                    torch.nn.init.constant_(self.cnn3.bias, 0)

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

                torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
                self.conv.weight.data *= 1e-2
                if self.conv.bias is not None:
                    torch.nn.init.constant_(self.conv.bias, 0)

            def forward(self, x):
                return self.relu(self.conv(x) * self.beta + x)


        class ChannelAttention(torch.nn.Module):
            def __init__(self, in_planes, reduction=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
                self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
                
                self.fc1 = torch.nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False, padding_mode='reflect')
                self.relu1 = torch.nn.LeakyReLU(0.2, inplace=True)
                self.fc2 = torch.nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False, padding_mode='reflect')
                
                self.sigmoid = torch.nn.Sigmoid()

                torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
                self.fc1.weight.data *= 1e-2
                torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
                self.fc2.weight.data *= 1e-2

            def forward(self, x):
                avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
                max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
                out = avg_out + max_out
                return self.sigmoid(out)

        class SpatialAttention(torch.nn.Module):
            def __init__(self, kernel_size=5):
                super(SpatialAttention, self).__init__()
                self.conv1 = torch.nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False, padding_mode='reflect')
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avg_out, max_out], dim=1)
                x = self.conv1(x)
                return self.sigmoid(x)

        class CBAMResBlock(torch.nn.Module):
            def __init__(self, in_channels, out_channels, stride=1, reduction=16, downsample=None):
                super(CBAMResBlock, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, padding_mode='reflect')
                self.relu = torch.nn.LeakyReLU(0.2, inplace=True)
                self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode='reflect')
                
                self.ca = ChannelAttention(out_channels, reduction)
                self.sa = SpatialAttention()

                self.beta = torch.nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)    

                self.downsample = downsample
                if stride != 1 or in_channels != out_channels:
                    self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True, padding_mode='reflect')

                torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
                self.conv1.weight.data *= 1e-2
                if self.conv1.bias is not None:
                    torch.nn.init.constant_(self.conv1.bias, 0)


            def forward(self, x):
                # residual = x if self.downsample is None else self.downsample(x)
                
                # ca_out =  self.relu(self.conv1(x * self.ca(x)) * self.beta1 + x)
                sa_out =  self.relu(self.conv2(x * self.sa(x)) * self.beta + x)

                return sa_out


        class Flownet(Module):
            def __init__(self, in_planes, c=64):
                super().__init__()
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c//2, 3, 2, 1),
                    conv(c//2, c, 3, 2, 1),
                    )
                self.convblock = torch.nn.Sequential(
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                    CBAMResBlock(c, c),
                )
                self.lastconv = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
                    torch.nn.PixelShuffle(2)
                )

            def forward(self, img0, img1, f0, f1, timestep, mask, flow, scale=1):
                timestep = (img0[:, :1].clone() * 0 + 1) * timestep
                
                if flow is None:
                    x = torch.cat((img0, img1, f0, f1, timestep), 1)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                else:
                    warped_img0 = warp(img0, flow[:, :2])
                    warped_img1 = warp(img0, flow[:, :2])
                    warped_f0 = warp(f0, flow[:, :2])
                    warped_f1 = warp(f1, flow[:, 2:4])
                    x = x = torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1)
                    x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
                    flow = torch.nn.functional.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
                    x = torch.cat((x, flow), 1)

                feat = self.conv0(x)
                feat = self.convblock(feat)
                tmp = self.lastconv(feat)
                tmp = torch.nn.functional.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
                flow = tmp[:, :4] * scale
                mask = tmp[:, 4:5]
                conf = tmp[:, 5:6]
                return flow, mask, conf

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

                flow, mask, conf = self.block0(img0, img1, f0, f1, timestep, None, None, scale=scale[0])

                flow_list[0] = flow.clone()
                mask_list[0] = torch.sigmoid(mask.clone())
                merged[0] = warp(img0, flow[:, :2]) * mask_list[0] + warp(img1, flow[:, 2:4]) * (1 - mask_list[0])

                for iteration in range(iterations):
                    flow_d, mask, conf = self.block1(
                        img0, 
                        img1,
                        f0,
                        f1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[1]
                    )
                    flow = flow + flow_d

                flow_list[1] = flow.clone()
                mask_list[1] = torch.sigmoid(mask.clone())
                merged[1] = warp(img0, flow[:, :2]) * mask_list[1] + warp(img1, flow[:, 2:4]) * (1 - mask_list[1])

                for iteration in range(iterations):
                    flow_d, mask, conf = self.block2(
                        img0, 
                        img1,
                        f0,
                        f1,
                        timestep,
                        mask,
                        flow, 
                        scale=scale[2]
                    )
                    flow = flow + flow_d

                flow_list[2] = flow.clone()
                mask_list[2] = torch.sigmoid(mask.clone())
                merged[2] = warp(img0, flow[:, :2]) * mask_list[2] + warp(img1, flow[:, 2:4]) * (1 - mask_list[2])

                for iteration in range(iterations):
                    flow_d, mask, conf = self.block3(
                        img0, 
                        img1,
                        f0,
                        f1,
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
            'name': 'Flownet4_v002h',
            'file': 'flownet4_v002h.py',
            'ratio_support': True
        }
        return info

    @staticmethod
    def get_name():
        return 'TWML_Flownet_v002h'

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