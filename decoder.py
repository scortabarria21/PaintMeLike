from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
from preprocess import load_tfrec_data

loss_tracker = tf.keras.metrics.Mean(name="loss")
mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")

class NeuralStyleTransferModel(tf.keras.Model):
    def __init__(self, content_layers, style_layers):
        super(NeuralStyleTransferModel, self).__init__()
        self.content_layers = content_layers
        self.style_layers = style_layers
        # I think the encoder should output something like (BATCH_SIZE, 128, 128, 64)
        self.decoder = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', padding='SAME'),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', padding='SAME'),
            tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', padding='SAME'),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', padding='SAME'),
            tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear'),
            tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), activation='relu', padding='SAME'),
            tf.keras.layers.Conv2DTranspose(3, kernel_size=(3, 3), activation=None, padding='SAME'),
        ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    
    def call(self, content_img, style_img):
        content_outputs = self.content_layers(content_img) # Encoding of content image
        style_outputs = self.style_layers(style_img) # Encoding of style image
        normalized_content = adaIn(content_outputs, style_outputs)
        return self.decoder(normalized_content)





def style_and_content_loss(content_pred, content_true, style_true, content_model, style_model, l=1.5):
    pred_content_feats = content_model(content_pred)
    pred_style_feats = style_model(content_pred)
    pred_gram = gram_matrix(pred_style_feats)

    true_content_feats = content_model(content_true)
    true_style_feats = style_model(style_true)
    true_gram = gram_matrix(true_style_feats)

    content_loss, style_loss = 0, 0

    for pred, true in zip(pred_content_feats, true_content_feats):
        content_loss += tf.reduce_mean(tf.square(pred - true))
    for pred, true in zip(pred_gram, true_gram):
        style_loss += tf.reduce_mean(tf.square(pred - true))
    style_loss *= 1/len(true_gram)
    return content_loss + l*style_loss

def adaIn(content, style):
    style = tf.image.resize(style, tf.shape(content)[1:3])
    style_mean, style_variance = tf.nn.moments(style, axes=[1, 2], keepdims=True)
    content_mean, content_variance = tf.nn.moments(content, axes=[1, 2], keepdims=True)
    normalized_content = tf.nn.batch_normalization(content, content_mean, content_variance, style_mean, tf.sqrt(style_variance), 1e-5)
    stylized_content = tf.multiply(normalized_content, tf.sqrt(style_variance)) + style_mean
    return stylized_content


def gram_matrix(input):
    """Returns gram matrix of an input."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input, input)
    input_shape = tf.shape(input)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result/(num_locations)

