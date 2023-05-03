from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
from preprocess import load_and_preprocess_jpeg, load_intel_data, load_tfrec_data

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
            tf.keras.layers.BatchNormalization(),
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
            tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=2, activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        res_out = self.residual_blocks(encoded)
        out = self.decoder(res_out)
        return out

    def loss(self, input, output, content_weight=1, style_weight=5, reconstruction_weight=0, total_var_weight=1e-5):
        input_scaled = input * 255.0
        style_target_preprocessed = self.style_target * 255.0
        output_scaled = output * 255.0
        style_target_preprocessed = tf.keras.applications.vgg19.preprocess_input(style_target_preprocessed)
        input_scaled = tf.keras.applications.vgg19.preprocess_input(input_scaled)
        output_scaled = tf.keras.applications.vgg19.preprocess_input(output_scaled)
        pred_content_feats = self.content_layers(output_scaled) # List of content outputs
        pred_gram = [gram_matrix(out) for out in self.style_layers(output_scaled)]

        true_content_feats = self.content_layers(input_scaled)
        true_gram = [gram_matrix(out) for out in self.style_layers(style_target_preprocessed)]

        content_loss, style_loss = 0, 0

        for pred, true in zip(pred_content_feats, true_content_feats):
            assert pred.shape == true.shape, "Content features have incompatible shapes"
            content_loss += tf.reduce_mean(tf.square(pred - true),)
        for pred, true in zip(pred_gram, true_gram):
            assert pred.shape == true.shape, "Style features have incompatible shapes"
            style_loss += tf.reduce_mean(tf.square(pred - true))
        style_loss *= 1.0/len(true_gram)

        reconstruction_loss = tf.reduce_mean(tf.square(input_scaled - output_scaled))
        total_variation_loss = tf.reduce_mean(tf.image.total_variation(output_scaled))

        return tf.reduce_mean(content_weight * content_loss + style_weight * style_loss + reconstruction_weight * reconstruction_loss + total_var_weight * total_variation_loss)


    

def gram_matrix(input_tensor):
    shape = tf.shape(input_tensor)
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    features = tf.reshape(input_tensor, [batch_size, -1, channels])
    gram = tf.matmul(features, features, transpose_a=True)
    num_elements = tf.cast(height * width, dtype=tf.float32)
    gram = gram / num_elements
    return gram

def train_step(model: ImageTransformer, data):
    total_loss = 0
    count = 0
    for image in data:
        count += 1
        with tf.GradientTape() as tape:
            output = model(image)
            # a = content feat weight, b = style weight, c = reconstruction weight, d = total variation loss
            # good arrangement: 6500, 0.00125, 1e-5
            loss = model.loss(image, output, content_weight=6500, style_weight=0.00125, total_var_weight=1e-5)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        total_loss += loss
        if count % 1000 == 0:
            print(f"{count} images read | Avg loss: {total_loss / count}")
            plt.imshow(image[0])
            plt.show()
            plt.imshow(output[0])
            plt.show()
    return total_loss


if __name__ == '__main__':
    tf.random.set_seed(42)
    image_data = load_tfrec_data('photo_tfrec', 1)
    style_target_unexpanded = load_and_preprocess_jpeg('monet.jpeg')
    style_target = tf.broadcast_to(tf.expand_dims(style_target_unexpanded, axis=0), (1, 256, 256, 3))

    content_layers_vgg = ['block4_conv2'] 
    style_layers_vgg = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1', 
                    'block4_conv1', 
                    'block5_conv1']

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False # we don't want to try to train it ourselves
    vgg_style_model = tf.keras.Model(inputs=[vgg.input], outputs=[vgg.get_layer(layer).output for layer in style_layers_vgg])
    vgg_content_model = tf.keras.Model(inputs=[vgg.input], outputs=[vgg.get_layer(layer).output for layer in content_layers_vgg])

    it_model = ImageTransformer(style_target, vgg_style_model, vgg_content_model)

    lr = 1e-3
    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    epochs = 5
    loss_list = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} beginning")
        loss = train_step(it_model, image_data)
        loss_list.append(loss)