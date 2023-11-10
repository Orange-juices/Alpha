import tensorflow as tf
import numpy as np
import time
import os

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
IMAGE_CHANNELS = 3
NETWORK_DEPTH = 4

data_dir = os.getcwd() + "/data/fruits-360/"
train_dir = data_dir + "Training/"
validation_dir = data_dir + "Test/"

batch_size = 60
input_size = IMAGE_HEIGHT * IMAGE_WIDTH * NETWORK_DEPTH
num_classes = len(os.listdir(train_dir))
# probability to keep the values after a training iteration
dropout = 0.8

initial_learning_rate = 0.001
final_learning_rate = 0.00001
learning_rate = initial_learning_rate

# number of iterations to run the training
iterations = 75000
# number of iterations after we display the loss and accuracy
acc_display_interval = 1000
# default number of iterations after we save the model
save_interval = 1000
step_display_interval = 100
# use the saved model and continue training
useCkpt = False
# placeholder for probability to keep the network parameters after an iteration
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# -------------------- Write/Read TF record logic --------------------
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def decode_jpeg(self, image_data):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3  # 检查条件，不符合就终止程序
        assert image.shape[2] == 3
        return image

def write_image_data(dir_name, tfrecords_name):
    writer = tf.python_io.TFRecordWriter(tfrecords_name)
    coder = ImageCoder()
    image_count = 0
    index = -1
    classes_dict = {}

    for folder_name in os.listdir(dir_name):
        class_path = dir_name + '/' + folder_name + '/'
        index += 1
        classes_dict[index] = folder_name
        for image_name in os.listdir(class_path):
            image_path = class_path + image_name
            image_count += 1
            with tf.gfile.FastGFile(image_path, 'rb') as f:
                image_data = f.read()
                example = tf.train.Example(
                    features = tf.train.Features(
                        feature = {
                            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            'image_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(image_data)]))
                        }
                    )
                )
                writer.write(example.SerializeToString())
    writer.close()
    print(classes_dict)
    return image_count, classes_dict
def parse_single_example(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.image.decode_jpeg(features['image_raw'], channels=3)
    image = tf.reshape(image, [100, 100, 3])
    label = tf.cast(features['label'], tf.int32)
    return image, label
