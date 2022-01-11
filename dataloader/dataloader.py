import tensorflow_datasets as tfds

class DataLoader():
	"""Data Loader Class"""	
	
	@staticmethod
	def load_data(data_config):
		""" Loads dataset from path"""
		
		return tfds.load( data_config.path, with_info=data_config.load_with_info )