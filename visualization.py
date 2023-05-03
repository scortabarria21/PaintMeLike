import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from image_transformer import ImageTransformer
from preprocess import load_and_preprocess_jpeg

def parse_png(image_path):
  image_string = tf.io.read_file(image_path)
  image = tf.io.decode_png(image_string, channels=3)
  image = tf.cast(image, dtype=tf.float32) / 255.0
  image = tf.image.resize(image, [256, 256]) 
  image = tf.expand_dims(image, axis=0)  
  return image

def style_vs_output_plot(style_images, stylized_images):
    fig, axes = plt.subplots(2, 4)

    axes = axes.flatten()
    fig.subplots_adjust(hspace=-0.5)
    for i in range(8):
        if i == 4:
            ax = axes[i]
            ax.imshow(parse_png('sayles.png')[0])
            title = ax.set_title('Content')
            title.set_fontsize(10)
            ax.set_xticks([])
            ax.set_yticks([])
        if 0 < i < 4:  
            ax = axes[i]
            if i == 1:
                ax.set_ylabel('Style')
            ax.imshow(style_images[i - 1])
            ax.set_xticks([])
            ax.set_yticks([])
        if 4 < i < 8:  
            ax = axes[i]
            ax.imshow(stylized_images[i - 5])
            ax.set_xticks([])
            ax.set_yticks([])


    fig.delaxes(axes[0])

    plt.show()

def reconstruction_plot():
    it_model.load_weights('it_model_checkpoints/cp-reconstruction.ckpt')

    reconstructed = it_model(parse_png('images/duck.png'))
    fig, axes = plt.subplots(1, 2)

    fig.suptitle('Reconstruction Only')
    axes[1].imshow(reconstructed[0])
    axes[0].imshow(parse_png('images/duck.png')[0])

    axes[0].set_title('Original')
    axes[1].set_title('Reconstruction')

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()

    plt.show()

def style_only_plot():
    it_model.load_weights('it_model_checkpoints_best/cp-style.ckpt')

    fig, axes = plt.subplots(1, 2)

    fig.suptitle('Style Only')

    style_only = it_model(tf.expand_dims(load_and_preprocess_jpeg('images/starry_night.jpeg'), axis=0))

    axes[0].imshow(load_and_preprocess_jpeg('images/starry_night.jpeg'))
    axes[1].imshow(style_only[0])

    axes[0].set_title('Original')
    axes[1].set_title('Style Output')

    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    fig.tight_layout()


if __name__ == '__main__':
    ckpts = ['van_gogh_landscape', 'picasso', 'monet']

    it_model = ImageTransformer()

    stylized_images = []
    style_images = []


    for ckpt in ckpts:
        it_model.load_weights('it_model_checkpoints/cp-{}_1.ckpt'.format(ckpt))
        stylized_images.append(it_model(parse_png('sayles.png'))[0])
        style_images.append(load_and_preprocess_jpeg('{}.jpeg'.format(ckpt)))
    
    style_vs_output_plot(style_images, stylized_images)

