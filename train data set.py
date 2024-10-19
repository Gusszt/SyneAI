import tensorflow as tf
import os

# Paths to your TFRecord files
train_tfrecord = 'tfrecord/train/Letters.tfrecord'
test_tfrecord = 'tfrecord/test/Letters.tfrecord'
valid_tfrecord = 'tfrecord/valid/Letters.tfrecord'


# Function to parse the TFRecord and adjust the label
def _parse_function(proto):
    # Define the features to extract from the TFRecord
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.FixedLenFeature([], tf.int64)
    }

    # Parse the features from the proto
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    # Decode the image from the TFRecord
    image = tf.io.decode_jpeg(parsed_features['image/encoded'], channels=3)

    # Resize the image to a fixed size and normalize
    image = tf.image.resize(image, [128, 128])
    image = image / 255.0  # Normalize pixel values to [0, 1]

    # Adjust the label to be 0-indexed
    label = parsed_features['image/object/class/label'] - 1

    return image, label


# Function to load and parse the TFRecord dataset
def load_dataset(tfrecord_path):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)  # Parse each example
    return dataset


# Load the datasets
train_dataset = load_dataset(train_tfrecord)
valid_dataset = load_dataset(valid_tfrecord)
test_dataset = load_dataset(test_tfrecord)

# Batch and shuffle the dataset
train_dataset = train_dataset.shuffle(1000).batch(32)
valid_dataset = valid_dataset.batch(32)
test_dataset = test_dataset.batch(32)

# Define your model (example)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')  # 26 output classes for A-Z
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=valid_dataset, epochs=10)

# Ensure the 'Model' directory exists
if not os.path.exists('Model'):
    os.makedirs('Model')

# Save the model to the 'Model' directory
model.save('Model/asl_model.h5')  # Save the model as 'asl_model.h5' in the 'Model' folder
