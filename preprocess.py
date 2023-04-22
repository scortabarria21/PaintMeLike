import tensorflow as tf
import os


def parse_tfrecord(example):
    """Parses a single image"""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
    }
    parsed_example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(parsed_example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [256, 256])
    return image


def load_data(filepath):
    """Loads the data given a filepath"""
    directory = filepath
    files = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            files.append(f)


    file_list_dataset = tf.data.Dataset.from_tensor_slices(files)

    data = file_list_dataset.interleave(
        lambda filename: tf.data.TFRecordDataset(filename).map(parse_tfrecord),
    )

    return data

if __name__ == '__main__':
    image_dataset = load_data('data/photo_tfrec')
    style_dataset = load_data('data/monet_tfrec')
    for image in image_dataset.take(5):
        print(image.shape)
