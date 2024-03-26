
class Model:
	def __init__(self, status = dict(), torch = None):
		if torch is None:
			import torch
		Module = torch.nn.Module

		class Conv2d(Module):
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
				# torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='selu')
				# torch.nn.init.xavier_uniform_(self.conv1.weight, gain=torch.nn.init.calculate_gain('selu'))
				# torch.nn.init.dirac_(self.conv1.weight)

			def forward(self,x):
				x = self.conv1(x)
				return x

		class Encoder(Module):
			def __init__(self):
				super().__init__()
				self.cnn0 = torch.nn.Conv2d(3, 32, 3, 2, 1, padding_mode = 'reflect', bias=False)
				self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode = 'reflect', bias=False)
				self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1, padding_mode = 'reflect', bias=False)
				self.cnn3 = torch.nn.ConvTranspose2d(32, 8, 4, 2, 1)
				self.relu = torch.nn.SELU(inplace = True)

			def forward(self, x):
				x = self.cnn0(x)
				x = self.relu(x)
				x = self.cnn1(x)
				x = self.relu(x)
				x = self.cnn2(x)
				x = self.relu(x)
				x = self.cnn3(x)
				return x

		self.model = Encoder
		self.training_model = Encoder

	@staticmethod
	def get_name():
		return 'TWML_Encoder_v001'
	
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
		channels = 8
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
