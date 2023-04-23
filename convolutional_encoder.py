from __future__ import absolute_import
from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
from preprocess import load_intel_data

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ConvolutionalEncoder(tf.keras.Model):
    """A convolutional neural network built to classify objects. 
    The internal representations to be developed by the convolutional layers here will be used as encodings
    of the images.

    We are training this on the intel dataset.
    """
    def __init__(self):
        super(ConvolutionalEncoder, self).__init__()
        self.num_classes = 6
        self.model = tf.keras.Sequential(layers=[
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv32_1'),
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv32_2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv64_1'),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv64_2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv128_1'),
            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='SAME', name='conv128_2'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        
            tf.keras.layers.Flatten(),
            
            tf.keras.layers.Dense(256, activation='relu', name='dense_1'),
            tf.keras.layers.Dense(256, activation='relu', name='dense_2'),
            tf.keras.layers.Dense(self.num_classes, activation='softmax', name='classification')
        
        ])



    def call(self, inputs):
        out = self.model(inputs)
        return out


def main():
    # Ignore this, we are running it in the notebook
    batch_size = 32
    train_dataset = load_intel_data('data/intel_images/seg_train', batch_size)
    test_dataset = load_intel_data('data/intel_images/seg_test', batch_size)
    epochs = 10
    model = ConvolutionalEncoder()
    lr = 1e-3

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy']
    )

    model.fit(
        train_dataset,
        epochs=epochs,
        validation_split=0.1,
        callbacks=None
    )


if __name__ == '__main__':
    main()