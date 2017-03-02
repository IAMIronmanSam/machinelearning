# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data",one_hot=True)

n_nodes_layer1 = 500

n_nodes_layer2 = 500

n_nodes_layer3 = 500

n_classes = 10
batch_size = 100

# height * width
x = tf.placeholder('float',[None, 784])
y = tf.placeholder('float',[None, 784])

def model(data):
    layer_1 = {'weights':tf.Variable(tf.random_normal([784, n_nodes_layer1])),
                'biases':tf.Variable(tf.random_normal(n_nodes_layer1))}

    layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_layer1, n_nodes_layer2])),
                'biases':tf.Variable(tf.random_normal(n_nodes_layer2))}

    layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_layer2, n_nodes_layer3])),
                'biases':tf.Variable(tf.random_normal(n_nodes_layer3))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_layer3, n_classes])),
                'biases':tf.Variable(tf.random_normal(n_classes))}

    # (input_data * weights) + biases

    neuron1 = tf.add(tf.matmul(data,layer_1['weights']), layer_1['biases'])
    neuron1 = tf.nn.relu(neuron1)

    neuron2 = tf.add(tf.matmul(data,layer_2['weights']), layer_2['biases'])
    neuron2 = tf.nn.relu(neuron2)

    neuron3 = tf.add(tf.matmul(data,layer_3['weights']), layer_3['biases'])
    neuron3 = tf.nn.relu(neuron3)

    output = tf.matmul(neuron3, output_layer['weights'] + output_layer['biases'])

    return output

def train(x):
    prediction = model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 5
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in hm_epochs:
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                x, y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: x, y: y})
                epoch_loss += c
            print('Epoch',epoch, 'completed out of', hm_epochs, 'loss:',epoch_loss)

    correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train(x)
