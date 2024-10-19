import tensorflow as tf


# Inspect the first example from the TFRecord
def inspect_tfrecord(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)

    for raw_record in raw_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        print(example)


# Replace 'path/to/train/Letters.tfrecord' with the actual path to your TFRecord
inspect_tfrecord('tfrecord/train/Letters.tfrecord')
