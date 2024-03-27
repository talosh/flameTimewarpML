class Model:
	def __init__(self, status = dict(), torch = None):
		if torch is None:
			import torch
		Module = torch.nn.Module

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
				torch.nn.SELU(inplace = True)
			)

		def warp(tenInput, tenFlow):
			backwarp_tenGrid = {}

			k = (str(tenFlow.device), str(tenFlow.size()))
			if k not in backwarp_tenGrid:
				tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
				tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
				backwarp_tenGrid[k] = torch.cat([ tenHorizontal, tenVertical ], 1).to(device)
				# end

			tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

			g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
			return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

		class ResConv(Module):
			def __init__(self, c, dilation=1):
				super().__init__()
				self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=False)
				self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
				self.relu = torch.nn.SELU(inplace = True)
				
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
					torch.nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
					torch.nn.PixelShuffle(2)
				)

			def forward(self, x, flow, scale=1):
				x = torch.nn.functional.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
				if flow is not None:
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
				# self.encode = Head()

			def forward(self, img0, img1, f0, f1, timestep=0.5, scale=[8, 4, 2, 1]):				
				if not torch.is_tensor(timestep):
					timestep = (img0[:, :1].clone() * 0 + 1) * timestep
				else:
					timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
				flow_list = []
				merged = []
				mask_list = []
				conf_list = []
				warped_img0 = img0
				warped_img1 = img1
				flow = None 
				stu = [self.block0, self.block1, self.block2, self.block3]
				flow = None
				for i in range(4):
					if flow is not None:
						flow_d, mask, conf = stu[i](torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask), 1), flow, scale=scale[i])
						flow = flow + flow_d
					else:
						flow, mask, conf = stu[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=scale[i])
					mask_list.append(mask)
					flow_list.append(flow)
					conf_list.append(conf)
					warped_img0 = warp(img0, flow[:, :2])
					warped_img1 = warp(img1, flow[:, 2:4])
					warped_f0 = warp(f0, flow[:, :2])
					warped_f1 = warp(f1, flow[:, 2:4])
					merged_student = (warped_img0, warped_img1)
					merged.append(merged_student)
				conf = torch.sigmoid(torch.cat(conf_list, 1))
				conf = conf / (conf.sum(1, True) + 1e-3)
				for i in range(4):
					mask_list[i] = torch.sigmoid(mask_list[i])
					merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
				return flow_list, mask_list, merged

		self.model = FlownetCas
		self.training_model = FlownetCas

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
