import tensorflow as tf
from tensorflow import keras

from random import shuffle
import glob
import cv2

import numpy as np
import matplotlib.pyplot as plt
import sys

from tkinter import Tk
from tkinter.filedialog import askopenfilename


# Keras.Flatten verandert een matrix naar een vector.

def create_model(train_images, train_labels, test_images, test_labels):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(240, 240)),
        keras.layers.Dense(960, activation=tf.nn.relu),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=10)

    val_loss, val_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy: ', val_acc)

    predictions = model.predict(test_images)

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 10
    num_cols = 8
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()

    Tk().withdraw()  # Simple GUI to select a file
    img = askopenfilename()
    img = prepare_image(img)
    img = (np.expand_dims(img, 0))

    predictions_single = model.predict(img)
    class_names = ['anomaly', 'clean']
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plot_image(0, predictions_single, test_labels, img)
    plt.subplot(1, 2, 2)
    plot_value_array(0, predictions_single, test_labels)
    _ = plt.xticks(range(2), class_names, rotation=45)
    plt.show()


def list_and_label_images(shuffle_data, training_data_percentage):
    training_path = 'training/anomalies_videos/samples_labeled/*.jpg'
    addrs = glob.glob(training_path)
    labels = [0 if '_ano' in addr else 1 for addr in addrs]

    # shuffle images to get different training
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(c)

    percentage_train = training_data_percentage / 100
    # percentage_val = (validation_percentage + training_data_percentage) / 100

    train_addrs = addrs[0:int(percentage_train * len(addrs))]
    train_labels = labels[0:int(percentage_train * len(labels))]

    # val_addrs = addrs[int(percentage_train*len(addrs)):int(percentage_val*len(addrs))]
    # val_labels = labels[int(percentage_train * len(labels)):int(percentage_val * len(labels))]

    test_addrs = addrs[int(percentage_train * len(addrs)):]
    test_labels = labels[int(percentage_train * len(labels)):]

    return train_addrs, train_labels, test_addrs, test_labels


def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write_tfrecords_train(filename):
    train_filename = filename  # address to save the TFRecords file
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(train_filename)
    for i in range(len(train_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Train data: {}/{}'.format(i, len(train_addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(train_addrs[i])
        label = train_labels[i]
        # Create a feature
        feature = {'train/label': _int64_feature(label),
                   'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def write_tfrecords_test(filename):
    test_filename = filename  # address to save the TFRecords file
    writer = tf.python_io.TFRecordWriter(test_filename)
    for i in range(len(test_addrs)):
        # print how many images are saved every 1000 images
        if not i % 1000:
            print('Test data: {}/{}'.format(i, len(test_addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(test_addrs[i])
        label = test_labels[i]
        # Create a feature
        feature = {'test/label': _int64_feature(label),
                   'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
    writer.close()
    sys.stdout.flush()


def plot_image(i, predictions_array, true_label, img):
    class_names = ['anomaly', 'clean']
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(2), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def prepare_image(image_source):
    image = cv2.imread(image_source)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image2 = np.array(gray / 255)
    # b, g, r = cv2.split(image)
    return image2


def create_image_array(addrs_list):
    image_array = []
    for i in range(len(addrs_list)):
        image_array.append(prepare_image(addrs_list[i]))
    image_array = np.array(image_array)
    return image_array


train_addrs, train_labels, test_addrs, test_labels = list_and_label_images(False, 80)
# print(train_addrs[0])
# print(create_image_array(train_addrs))
# print(train_labels)
# write_tfrecords_train('train.tfrecords')
# write_tfrecords_test('test.tfrecords')

train_images = create_image_array(train_addrs)
test_images = create_image_array(test_addrs)
create_model(train_images, train_labels, test_images, test_labels)
