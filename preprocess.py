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


def load_tfrec_data(filepath, batch_size):
    """Loads the tfrec data given a filepath"""
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
    return data.batch(batch_size)

def load_and_preprocess_jpeg(filepath):
    """Loads the jpeg data given a filepath"""
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [256, 256])
    return image


def load_jpeg_data(filepath, batch_size):
    directory = filepath
    file_names = os.listdir(directory)
    file_paths = [os.path.join(directory, file_name) for file_name in file_names]
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(load_and_preprocess_jpeg)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset.batch(batch_size)


def load_intel_data(filepath, batch_size):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
    filepath,
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=(256, 256),
    shuffle=True,
    seed=42,
    )
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    # rescale to [0, 1]
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))
    return dataset