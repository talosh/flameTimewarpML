class Model:
	def __init__(self, status = dict(), torch = None):
		if torch is None:
			import torch
		Module = torch.nn.Module

		class ResidualBlock(nn.Module):
			def __init__(self, in_planes, planes, norm_fn='group', stride=1):
				super().__init__()
		
				self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
				self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, padding=1)
				# self.relu = torch.nn.ReLU(inplace=True)
				self.relu = torch.nn.SELU(inplace=True)

				num_groups = planes // 8

				if norm_fn == 'group':
					self.norm1 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=planes)
					self.norm2 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=planes)
					# if not stride == 1:
					self.norm3 = torch.nn.GroupNorm(num_groups=num_groups, num_channels=planes)
				
				elif norm_fn == 'batch':
					self.norm1 = torch.nn.BatchNorm2d(planes)
					self.norm2 = torch.nn.BatchNorm2d(planes)
					# if not stride == 1:
					self.norm3 = torch.nn.BatchNorm2d(planes)
				
				elif norm_fn == 'instance':
					self.norm1 = torch.nn.InstanceNorm2d(planes)
					self.norm2 = torch.nn.InstanceNorm2d(planes)
					# if not stride == 1:
					self.norm3 = torch.nn.InstanceNorm2d(planes)

				elif norm_fn == 'none':
					self.norm1 = torch.nn.Sequential()
					self.norm2 = torch.nn.Sequential()
					# if not stride == 1:
					self.norm3 = torch.nn.Sequential()

				# if stride == 1:
				#     self.downsample = None
				#
				# else:
				self.downsample = torch.nn.Sequential(
					torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

			def forward(self, x):
				y = x
				y = self.relu(self.norm1(self.conv1(y)))
				y = self.relu(self.norm2(self.conv2(y)))

				if self.downsample is not None:
					x = self.downsample(x)
					
				return self.relu(x+y)


		self.model = MultiResUnet_MemOpt
		self.training_model = MultiResUnet

	@staticmethod
	def get_name():
		return 'SparseNet_v001'
	
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
		channels = 3
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