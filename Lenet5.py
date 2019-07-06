from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten

mnist = input_data.read_data_sets("/datasets/ud730/mnist", reshape=False)
X_train, y_train = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test = mnist.test.images, mnist.test.labels

assert (len(X_train) == len(y_train))
assert (len(X_validation) == len(y_validation))
assert (len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))

# Pad images with 0s
X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1, 1))
plt.imshow(image, cmap="gray")
plt.show()

print(y_train[index])
X_train, y_train = shuffle(X_train, y_train)
EPOCHS = 10
BATCH_SIZE = 128
meanweight = 0.0
standardDev = 0.1
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 6], mean=meanweight, stddev=standardDev)),
    'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16], mean=meanweight, stddev=standardDev)),
    'wFull1': tf.Variable(tf.random_normal([400, 120], mean=meanweight, stddev=standardDev)),
    'wFull2': tf.Variable(tf.random_normal([120, 84], mean=meanweight, stddev=standardDev)),
    'wFull3': tf.Variable(tf.random_normal([84, 10], mean=meanweight, stddev=standardDev))
}

biases = {
    'biC1': tf.Variable(tf.random_normal([6])),
    'biC2': tf.Variable(tf.random_normal([16])),
    'biFlatten': tf.Variable(tf.random_normal([400])),
    'biFullConnected1': tf.Variable(tf.random_normal([120])),
    'biFullConnected2': tf.Variable(tf.random_normal([84])),
    'biFullConnected3': tf.Variable(tf.random_normal([10]))
}


def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)  # why do we need relu?


def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    convL1 = conv2d(x, weights['wc1'], biases['biC1'])
    # TODO: Activation.
    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    convL1 = maxpool2d(convL1, k=2)
    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    convL2 = conv2d(convL1, weights['wc2'], biases['biC2'])
    # TODO: Activation.
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    convL2 = maxpool2d(convL2, k=2)
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(convL2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fL1 = tf.add(tf.matmul(fc1, weights['wFull1']), biases['biFullConnected1'])
    # TODO: Activation.
    fL1 = tf.nn.relu(fL1)
    # fL1 = tf.nn.dropout(fL1,)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fL2 = tf.add(tf.matmul(fL1, weights['wFull2']), biases['biFullConnected2'])
    # TODO: Activation.
    fL2 = tf.nn.relu(fL2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fL2, weights['wFull3']), biases['biFullConnected3'])
    return logits


x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)
rate = 0.001

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
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    # conv1 = sess.graph.get_tensor_by_name('weights')
    # outputFeatureMap(X_train, conv1)
