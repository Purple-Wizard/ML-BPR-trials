import tensorflow as tf
from keras.layers import Conv2D, Layer

class HashingLayer(Layer):
    def __init__(self, output_channels, **kwargs):
        super(HashingLayer, self).__init__(**kwargs)
        self.output_channels = output_channels

    def build(self, input_shape):
        self.conv = Conv2D(self.output_channels, (1, 1), strides=(1, 1), padding='same')
        super(HashingLayer, self).build(input_shape)

    def call(self, x):
        return self.conv(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.output_channels)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "output_channels": self.output_channels}