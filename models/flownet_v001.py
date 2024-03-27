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

		class ResConv(Module):
			def __init__(self, c, dilation=1):
				super().__init__()
				self.conv = torch.nn.Conv2d(c, c, 3, 1, dilation, dilation = dilation, groups = 1, padding_mode = 'reflect', bias=False)
				self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
				self.relu = torch.nn.SELU(inplace = True)
				
			def forward(self, x):
				return self.relu(self.conv(x) * self.beta + x)

		class FlownetBlock(Module):
			def __init__(self, in_planes, c=64):
				super().__init__()
				self.conv0 = torch.nn.Sequential(
					conv(in_planes, c//2, 3, 2, 1),
					conv(c//2, c, 3, 2, 1),
					)
				self.convblock = nn.Sequential(
					ResConv(c),
					ResConv(c),
					ResConv(c),
					ResConv(c),
					ResConv(c),
					ResConv(c),
					ResConv(c),
					ResConv(c),
				)
				self.lastconv = nn.Sequential(
					nn.ConvTranspose2d(c, 4*6, 4, 2, 1),
					nn.PixelShuffle(2)
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


		self.model = Flownet
		self.training_model = Flownet



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
