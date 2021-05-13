# -*- coding =utf-8 -*-
# @Project : DataMining
# @Time : 13:23
# @Author : XianweiCao
# @Package :
# @File : classify.py
# @Software: PyCharm Professional

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

dropout = 0.85

mnist = input_data.read_data_sets("MNIST_data/", one_hot='true')

batch_x, batch_y = mnist.test.next_batch(10)

plt.imshow(batch_x[0].reshape(28, 28))
plt.show()

n_input = 784
# 共10类
n_classes = 10


def classify():
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph('./model.ckpt.data-00000-of-00001.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    # 获取权重
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("n_input:0")
    y = graph.get_tensor_by_name("n_classes:0")
    # x = tf.placeholder(tf.float32, [None, n_input], name='n_input')
    # y = tf.placeholder(tf.float32, [None, n_classes], name='n_classes')

    # Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    print("Model restored.")
    result = sess.run(y, feed_dict={x: [None, batch_x[0]]})


# classify()


def load_model():
    with tf.Session() as sess:
        # saver = tf.train.import_meta_graph('./model.ckpt.data-00000-of-00001.meta')
        saver = tf.train.import_meta_graph('./model/model.ckpt.meta')

        saver.restore(sess, tf.train.latest_checkpoint('./model/'))

        graph = tf.get_default_graph()
        tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
        for i in tensor_name_list:
            print(i)
        x = graph.get_tensor_by_name('n_input:0')  # 获取输入变量（占位符，由于保存时未定义名称，tf自动赋名称“Placeholder”）
        y = graph.get_tensor_by_name('n_classes:0')  # 获取输出变量
        keep_prob = graph.get_tensor_by_name('Placeholder:0')  # 获取dropout的保留参数

        pred = graph.get_tensor_by_name('Add_1:0')  # 获取网络输出值
        # 定义评价指标
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
        correct = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.Session() as sess:
            # saver = tf.train.import_meta_graph('./model.ckpt.data-00000-of-00001.meta')
            saver = tf.train.import_meta_graph('./model/model.ckpt.meta')

            saver.restore(sess, tf.train.latest_checkpoint('./model/'))
            print('finish loading model!')
            # test
            image = batch_x[0]
            test_out = sess.run([accuracy, loss], feed_dict={x: [1, image], y: 3, keep_prob: dropout})
            print(test_out)


load_model()
