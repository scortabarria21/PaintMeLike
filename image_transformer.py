from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
from preprocess import load_intel_data

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='SAME', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='SAME', activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        output = inputs + x
        return output


class ImageTransformer(tf.keras.Model):
    def __init__(self, style_target, style_layers, content_layers):
        """Image transformer model. Takes in one style target, the VGG style layers, and VGG content layers.
        """
        super(ImageTransformer, self).__init__()
        self.style_target = style_target
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.encoder = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=2, activation='relu'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=(2, 2), strides=2, activation='relu')
        ])

        self.residual_blocks = tf.keras.Sequential(layers=[
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock(),
            ResidualBlock()
        ])


        self.decoder = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=2, activation='relu'),
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=2, activation='sigmoid')
        ])


    def call(self, x):
        encoded = self.encoder(x)
        res_out = self.residual_blocks(encoded)
        out = self.decoder(res_out)
        return out

    def loss(self, input, output, a=1000, b=0.08, c=0, d=0.03):
        pred_content_feats = self.content_layers(output) # List of content outputs
        pred_gram = [gram_matrix(out) for out in self.style_layers(output)]

        true_content_feats = self.content_layers(input)
        true_gram = [gram_matrix(out) for out in self.style_layers(self.style_target)]

        content_loss, style_loss = 0, 0

        for pred, true in zip(pred_content_feats, true_content_feats):
            content_loss += tf.reduce_mean(tf.square(pred - true))
        for pred, true in zip(pred_gram, true_gram):
            style_loss += tf.reduce_mean(tf.square(pred - true))
        style_loss *= 1/len(true_gram)

        reconstruction_loss = tf.reduce_mean(tf.square(input - output))
        total_variation_loss = tf.image.total_variation(output)

        return a * content_loss + b * style_loss + c * reconstruction_loss + d * total_variation_loss


    

def gram_matrix(input):
    """Returns gram matrix of an input."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input, input)
    input_shape = tf.shape(input)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)