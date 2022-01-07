"""Unet model"""

# standard library

# internal
from .base_model import BaseModel
from dataloader.dataloader import DataLoader

# external
import tensorflow as tf

class UNet(BaseModel)
	"""Unet Model class. Contains functionality for building, training and evaluating the model"""
	
	def __init__(self, config):
		super().__init__(config)
		self.base_model = tf.keras.applications.MobileNetV2(
			input_shape=self.config.model.input, include_top = False)
			
	def load_data(self):
		self.dataset, self.info = DataLoader().load_data(self.config.data )
		self._preprocess_data()
		
	def build(self):
		
		layer_names = [
			"block_1_expand_relu", #64x64
			"block_3_expand_relu", #32x32
			"block_6_expand_relu", #16x16
			"block_13_expand_relu", #8x8
			"block_16_project", #4x4
			]
		layers = [self.base_model.get_layer(name).output for name in layer_names
		
		self.model = tf.keras.Model(input = inputs, outputs = x )
		
	def train(self):
		self.model.compile(
			optimizer = self.config.train.optimizer.type
			loss = tf.keras.losses.SparseCategoricalCrossentropy(
				from_logits = True
				),
			metrics = self.config.train.metrics
		)
			
		model_history = self.model.fit(
			self.train_dataset,
			epochs = self.epcohs
			steps_per_epoch = self.steps_per_epoch
			validation_steps = self.validation_steps
			validation_data = self.validation_data
		)
		
		return model_history.history['loss'], model_history.historl['val_loss']
	
	def evaluate(self):
		predictions = []
		for image, mask in self.dataset.take(1):
			predictions.append(self.model.predict(image) )
			
		return predictions
	
	def _preprocess_data(self):
		""" Splits into training and test and set training parameters"""
		train = self.dataset['train'].map(
			self.load_image_train,
			num_parallel_calls = tf.data.experimental.AUTOTUNE
		)
		
		self.train_dataset = train
			.cache()
			.shuffle(self.buffer_size)
			.batch(self.batch_size)
			.repeat()
			
		self.train_dataset = self.train_dataset.prefetch(
			buffer_size = tf.data.experimental.AUTOTUNE
		)
		
		self.test_dataset = test.batch(self.batch_size)
		
		
		
	def _set_training_parameters(self):
		pass
		
	def _normalize(self, input_image, input_mask):
		""" Normalise input image
		
		Args:
			input_image (tf.image): The input image
			input_mask (int): The image mask
		
		Returns:
			input_image (tf.image): The normalized input image
			input_mask (int): The new image mask
		"""
		
		input_image = tf.cast(input_image, tf.float32) / 255.0
		input_mask -= 1
		return input_image, input_mask
		
	def _load_image_train(self, datapoint):
		""" Loads and preprocess a single training image """
		input_image = tf.image.resize(
			datapoint['image'], (self.image_size, self.image_size )
		)
		
		input_mask = tf.iage.resize(
			datapoint['segmentation_mask'], (self.image_size, self.image_size )
		)
		
		if tf.random.uniform(()) > 0.5:
			input_image = tf.image.flip_left_right(input_image)
			input_mask = tf.image.flip_left_right(input_mask)
		
		input_image, input_mask = self.normalize( input_image, input_mask )
		
		return input_image, input_mask
		
	def _load_image_test(self, datapoint):
		""" Loads and preprocesss a single test image """
		
		input_image = tf.image.resize( datapoint['image'], (self.image_size, self.image_size ) )
		
		input_mask = tf.image.resize( datapoint['segmentation_mask'], (self.image_size, self.image_size ) )
		
		input_image, input_mask = self.normalize( input_image, input_mask )
		
		return input_image, input_mask