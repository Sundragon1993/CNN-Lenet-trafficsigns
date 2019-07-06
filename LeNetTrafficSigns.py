from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from urllib.request import urlretrieve
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from zipfile import ZipFile
import math
# Load pickled data
import pickle

# def download(url, file):
#     if not os.path.isfile(file):
#         print("Download file... " + file + " ...")
#         urlretrieve(url, file)
#         print("File downloaded")
#
#
# download(
#     'https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip',
#     'data.zip')
# print("All the files are downloaded")
#
#
# only to load dataset from internet
# def uncompress_features_labels(dir):
#     if (os.path.isdir('traffic-data')):
#         print('Data extracted')
#     else:
#         with ZipFile(dir) as zipf:
#             zipf.extractall('traffic-signs-data')
#
#
# uncompress_features_labels('data.zip')

# def data_Files(mypath):
#     onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
#     print(onlyfiles)


# TODO: Fill this in based on where you saved the training and testing data
training_file = './traffic-signs-data/train.p'
validation_file = './traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
print("Data Loaded from pickle files!")

# Number of training examples
n_train = np.shape(X_train)[0]

# Number of validation examples
n_validation = np.shape(X_valid)[0]

# Number of testing examples.
n_test = np.shape(X_test)[0]

# What's the shape of an traffic sign image?
image_shape = np.shape(X_train[0])

# How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

print("Updated Image Shape: {}".format(X_train[0].shape))

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1, 1))
plt.imshow(image)
plt.show()
print(y_train[index])

X_train, y_train = shuffle(X_train, y_train)
EPOCHS = 20
BATCH_SIZE = 128
meanweight = 0.0
standardDev = 0.1
weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 3, 6], mean=meanweight, stddev=standardDev)),
    'wc2': tf.Variable(tf.truncated_normal([5, 5, 6, 16], mean=meanweight, stddev=standardDev)),
    'wFull1': tf.Variable(tf.truncated_normal([400, 120], mean=meanweight, stddev=standardDev)),
    'wFull2': tf.Variable(tf.truncated_normal([120, 84], mean=meanweight, stddev=standardDev)),
    'wFull3': tf.Variable(tf.truncated_normal([84, 43], mean=meanweight, stddev=standardDev))
}

biases = {
    'biC1': tf.Variable(tf.truncated_normal([6])),
    'biC2': tf.Variable(tf.truncated_normal([16])),
    'biFlatten': tf.Variable(tf.truncated_normal([400])),
    'biFullConnected1': tf.Variable(tf.truncated_normal([120])),
    'biFullConnected2': tf.Variable(tf.truncated_normal([84])),
    'biFullConnected3': tf.Variable(tf.truncated_normal([43]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)  # why do we need relu?


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


from tensorflow.contrib.layers import flatten


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True) + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 14x14x10.
    conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 10), mean=mu, stddev=sigma))
    conv3_b = tf.Variable(tf.zeros(10))
    conv3 = tf.nn.conv2d(conv1, conv3_W, strides=[1, 2, 2, 1], padding='VALID', use_cudnn_on_gpu=True) + conv3_b

    # Activation.
    conv3 = tf.nn.relu(conv3)

    # Layer 3: Convolutional. Input = 14x14x10. Output = 8x8x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 10, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv3, conv2_W, strides=[1, 1, 1, 1], padding='VALID', use_cudnn_on_gpu=True) + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 8x8x16. Output = 4x4x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 4x4x16. Output = 256.
    f = flatten(conv2)

    # Layer 4: Fully Connected. Input = 256. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(int(np.shape(f)[1]), 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(shape=120))
    fc1 = tf.matmul(f, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Introduce Dropout after first fully connected layer
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 5: Fully Connected. Input = 120. Output = 100.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 100), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(100))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 6: Fully Connected. Input = 100. Output = 84.
    fc4_W = tf.Variable(tf.truncated_normal(shape=(100, 84), mean=mu, stddev=sigma))
    fc4_b = tf.Variable(tf.zeros(84))
    fc4 = tf.matmul(fc2, fc4_W) + fc4_b

    # Activation.
    fc4 = tf.nn.relu(fc4)

    # Layer 7: Fully Connected. Input = 84. Output = 43.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    fc3 = tf.matmul(fc4, fc3_W) + fc3_b
    logits = fc3

    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)  # one hot encoding for output labels
keep_prob = tf.placeholder(
    tf.float32)  # defining the dropout probability after fully connected layer in the architecture
print('Variables initialized successfully')

rate = 0.0009  # learning rate

# defining various operations
logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# # returning Indices of the max element
# # as per the indices
#
# '''
#    [[ 0  3  8 13]
#     [12 11  2 11]
#     [ 5 13  8  3]
#     [12 15  3  4]]
#       ^  ^  ^  ^
#      12 15  8  13  - element
#      1  3   0  0   - indices
# '''
# print("\nIndices of Max element : ", geek.argmax(array, axis = 0))
#
#
# '''
#                             ELEMENT   INDEX
#    ->[[ 0  3  8 13]           13        3
#     ->[12 11  2 11]           12        0
#     ->[ 5 13  8  3]           13        1
#     ->[12 15  3  4]]          15        1
#
# '''
# print("\nIndices of Max element : ", geek.argmax(array, axis = 1))


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy, loss = sess.run([accuracy_operation, loss_operation],
                                  feed_dict={x: batch_x, y: batch_y, keep_prob: 1})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))  # getting the total loss to plot a graph later
    return total_accuracy / num_examples, total_loss / num_examples


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(X_train)
#
#     print("Training...")
#     print()
#     loss_Acc = []
#     for i in range(EPOCHS):
#         X_train, y_train = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
#
#         validation_accuracy, loss_acc = evaluate(X_valid, y_valid)
#
#         print("EPOCH {} ...".format(i + 1))
#         loss_Acc.append(loss_acc)
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
#     plt.plot(range(0, EPOCHS), loss_Acc)
#     plt.ylabel('loss')
#     plt.xlabel('Epochs')
#     plt.grid(True)
#     plt.show()
#     saver.save(sess, './trafficTest')
#     print("Model saved")
#
# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy[0]))

# Load the images and plot them here.
# Feel free to use as many code cells as needed.
import os
import matplotlib.image as mpimg
import cv2

my_images = []

for i, img in enumerate(os.listdir('./traffic-signs-real/new/')):
    image = cv2.imread('./traffic-signs-real/new/' + img)
    my_images.append(image)
    plt.figure()
    plt.xlabel(img)
    plt.imshow(image)
    plt.show()

my_images = np.asarray(my_images)

my_labels = [35, 29, 17, 27, 31]
# Check Test Accuracy

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    output_accuracy = evaluate(my_images, my_labels)
    print("Test Accuracy = {:.3f}".format(output_accuracy[0]))
