class Model:
    def __init__(self, status = dict(), torch = None):
        if torch is None:
            import torch
        Module = torch.nn.Module

        def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
                torch.nn.PReLU(out_planes)
            )

        def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes,
                                        kernel_size=4, stride=2, padding=1, bias=True),
                torch.nn.PReLU(out_planes)
            )

        # '''
        def warp(tenInput, tenFlow):
            input_device = tenInput.device
            tenInput = tenInput.detach().to(device=torch.device('cpu'))
            tenFlow = tenFlow.detach().to(device=torch.device('cpu'))

            backwarp_tenGrid = {}

            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device=tenInput.device)
                # end

            tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            result = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='reflection', align_corners=True)
            return result.detach().to(device=input_device)
        # '''

        '''
        def warp(tenInput, tenFlow):
            original_device = tenInput.device
            device = 'cpu'
            tenInput = tenInput.to(device=device)
            tenFlow = tenFlow.to(device=device)

            backwarp_tenGrid = {}
            k = (str(tenFlow.device), str(tenFlow.size()))
            if k not in backwarp_tenGrid:
                tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
                    1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
                tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
                    1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
                backwarp_tenGrid[k] = torch.cat(
                    [tenHorizontal, tenVertical], 1).to(device)

            tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                                tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

            g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
            result = torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
            # return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
            # cpu_g = g.to('cpu')
            # cpu_tenInput = tenInput.to('cpu')
            # cpu_result = torch.nn.functional.grid_sample(input=cpu_tenInput, grid=cpu_g, mode='bicubic', padding_mode='border', align_corners=True)
            return result.to(device = original_device)
        '''

        class Conv2(Module):
            def __init__(self, in_planes, out_planes, stride=2):
                super().__init__()
                self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
                self.conv2 = conv(out_planes, out_planes, 3, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        class ContextNet(Module):
            def __init__(self):
                c = 32
                super().__init__()
                self.conv0 = Conv2(3, c)
                self.conv1 = Conv2(c, c)
                self.conv2 = Conv2(c, 2*c)
                self.conv3 = Conv2(2*c, 4*c)
                self.conv4 = Conv2(4*c, 8*c)

            def forward(self, x, flow):
                x = self.conv0(x)
                x = self.conv1(x)
                flow = torch.nn.functional.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
                f1 = warp(x, flow)
                x = self.conv2(x)
                flow = torch.nn.functional.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                    align_corners=False) * 0.5
                f2 = warp(x, flow)
                x = self.conv3(x)
                flow = torch.nn.functional.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                    align_corners=False) * 0.5
                f3 = warp(x, flow)
                x = self.conv4(x)
                flow = torch.nn.functional.interpolate(flow, scale_factor=0.5, mode="bilinear",
                                    align_corners=False) * 0.5
                f4 = warp(x, flow)
                return [f1, f2, f3, f4]

        class IFBlock(Module):
            def __init__(self, in_planes, scale=1, c=64):
                super(IFBlock, self).__init__()
                self.scale = scale
                self.conv0 = torch.nn.Sequential(
                    conv(in_planes, c, 3, 2, 1),
                    conv(c, 2*c, 3, 2, 1),
                    )
                self.convblock = torch.nn.Sequential(
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                    conv(2*c, 2*c),
                )        
                self.conv1 = torch.nn.ConvTranspose2d(2*c, 4, 4, 2, 1)
                            
            def forward(self, x):
                if self.scale != 1:
                    x = torch.nn.functional.interpolate(
                        x, 
                        scale_factor=1. / self.scale,
                        mode="bilinear",
                        align_corners=False
                        )
                x = self.conv0(x)
                x = self.convblock(x)
                x = self.conv1(x)
                flow = x
                if self.scale != 1:
                    flow = torch.nn.functional.interpolate(
                        flow, 
                        scale_factor=self.scale, 
                        mode="bilinear",
                        align_corners=False
                        )
                return flow

        class IFNet(Module):
            def __init__(self):
                super(IFNet, self).__init__()
                self.block0 = IFBlock(6, scale=8, c=192)
                self.block1 = IFBlock(10, scale=4, c=128)
                self.block2 = IFBlock(10, scale=2, c=96)
                self.block3 = IFBlock(10, scale=1, c=48)

            def forward(self, x, UHD=False):
                if UHD:
                    x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)

                self.block0.to(device=torch.device('mps'))
                self.block1.to(device=torch.device('mps'))
                self.block2.to(device=torch.device('mps'))
                self.block3.to(device=torch.device('mps'))
                x = x.detach().to(device=torch.device('mps'))


                flow0 = self.block0(x)
                F1 = flow0
                F1_large = torch.nn.functional.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 2.0
                warped_img0 = warp(x[:, :3], F1_large[:, :2])
                warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
                flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
                F2 = (flow0 + flow1)
                F2_large = torch.nn.functional.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 2.0
                warped_img0 = warp(x[:, :3], F2_large[:, :2])
                warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
                flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
                F3 = (flow0 + flow1 + flow2)
                F3_large = torch.nn.functional.interpolate(F3, scale_factor=2.0, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 2.0
                warped_img0 = warp(x[:, :3], F3_large[:, :2])
                warped_img1 = warp(x[:, 3:], F3_large[:, 2:4])
                flow3 = self.block3(torch.cat((warped_img0, warped_img1, F3_large), 1))
                F4 = (flow0 + flow1 + flow2 + flow3)

                F4 = F4.detach().to(device=torch.device('cpu'))

                return F4, [F1, F2, F3, F4]

        class FusionNet(Module):
            def __init__(self):
                super(FusionNet, self).__init__()
                c = 32
                self.conv0 = Conv2(10, c)
                self.down0 = Conv2(c, 2*c)
                self.down1 = Conv2(4*c, 4*c)
                self.down2 = Conv2(8*c, 8*c)
                self.down3 = Conv2(16*c, 16*c)
                self.up0 = deconv(32*c, 8*c)
                self.up1 = deconv(16*c, 4*c)
                self.up2 = deconv(8*c, 2*c)
                self.up3 = deconv(4*c, c)
                self.conv = torch.nn.ConvTranspose2d(c, 4, 4, 2, 1)

            def forward(self, img0, img1, flow, c0, c1, flow_gt):
                warped_img0 = warp(img0, flow[:, :2])
                warped_img1 = warp(img1, flow[:, 2:4])
                if flow_gt == None:
                    warped_img0_gt, warped_img1_gt = None, None
                else:
                    warped_img0_gt = warp(img0, flow_gt[:, :2])
                    warped_img1_gt = warp(img1, flow_gt[:, 2:4])
                x = self.conv0(torch.cat((warped_img0, warped_img1, flow), 1))
                s0 = self.down0(x)
                s1 = self.down1(torch.cat((s0, c0[0], c1[0]), 1))
                s2 = self.down2(torch.cat((s1, c0[1], c1[1]), 1))
                s3 = self.down3(torch.cat((s2, c0[2], c1[2]), 1))
                x = self.up0(torch.cat((s3, c0[3], c1[3]), 1))
                x = self.up1(torch.cat((x, s2), 1))
                x = self.up2(torch.cat((x, s1), 1))
                x = self.up3(torch.cat((x, s0), 1))
                x = self.conv(x)
                return x, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt
            
        class FlownetModel(Module):
            def __init__(self):
                super().__init__()
                self.flownet = IFNet()
                self.contextnet = ContextNet()
                self.fusionnet = FusionNet()

            def forward(self, img0, gt, img1, f0, f1, timestep=0.5, scale=[8, 4, 2, 1]):
                imgs = torch.cat((img0, img1), 1)
                flow, _ = self.flownet(imgs, UHD=False)
                return self.predict(imgs, flow, training=False, UHD=False)

            def predict(self, imgs, flow, training=True, flow_gt=None, UHD=False):
                img0 = imgs[:, :3]
                img1 = imgs[:, 3:]
                if UHD:
                    flow = torch.nn.functional.interpolate(flow, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
                c0 = self.contextnet(img0, flow[:, :2])
                c1 = self.contextnet(img1, flow[:, 2:4])
                flow = torch.nn.functional.interpolate(flow, scale_factor=2.0, mode="bilinear",
                                    align_corners=False) * 2.0
                refine_output, warped_img0, warped_img1, warped_img0_gt, warped_img1_gt = self.fusionnet(
                    img0, img1, flow, c0, c1, flow_gt)
                res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
                mask = torch.sigmoid(refine_output[:, 3:4])
                merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
                pred = merged_img + res

                pred.to(device=torch.device('mps'))
                
                return [flow] * 4, [mask] * 4, [pred] * 4

            def load_old_model(self, path, rank=0):
                def convert(param):
                    return {
                        k.replace("module.", ""): v
                        for k, v in param.items()
                        if "module." in k
                    }
                    
                device = next(self.parameters()).device
                print (f'loading old model to device: {device}')
                # flownet_dict = torch.load('{}/flownet.pkl'.format(path), map_location=device)


                # '''
                self.flownet.load_state_dict(
                    convert(torch.load('{}/flownet.pkl'.format(path), map_location=device)))
                self.contextnet.load_state_dict(
                    convert(torch.load('{}/contextnet.pkl'.format(path), map_location=device)))
                self.fusionnet.load_state_dict(
                    convert(torch.load('{}/unet.pkl'.format(path), map_location=device)))
                # '''

        self.model = FlownetModel
        self.training_model = FlownetModel

    @staticmethod
    def get_info():
        info = {
            'name': 'Flownet2_v004',
            'file': 'flownet2_v004.py',
            'ratio_support': False
        }
        return info

    @staticmethod
    def get_name():
        return 'Flownet4_v008'

    @staticmethod
    def get_file_name():
        import os
        return os.path.basename(__file__)

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
        flownet.load_old_model(path)