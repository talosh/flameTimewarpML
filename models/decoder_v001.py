
class Model:
	def __init__(self, status = dict(), torch = None):
		if torch is None:
			import torch
		Module = torch.nn.Module

		class Decoder(Module):
			def __init__(self):
				super().__init__()
				self.cnn0 = torch.nn.Conv2d(8, 32, 3, 2, 1)
				self.cnn1 = torch.nn.Conv2d(32, 32, 3, 1, 1)
				self.cnn2 = torch.nn.Conv2d(32, 32, 3, 1, 1)
				self.cnn3 = torch.nn.ConvTranspose2d(32, 3, 4, 2, 1)
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
		return 'TWML_Decoder_v001'
	
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
