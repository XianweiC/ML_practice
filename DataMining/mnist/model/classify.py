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
import numpy as np

dropout = 0.85

mnist = input_data.read_data_sets("../MNIST_data/", one_hot='true')

batch_x, batch_y = mnist.test.next_batch(10)

n_input = 784
# 共10类
n_classes = 10

# from PIL import Image, ImageFilter

# def imageprepare():
#     im = Image.open('C:/Users/考拉拉/Desktop/4.png')  # 读取的图片所在路径，注意是28*28像素
#     plt.imshow(im)  # 显示需要识别的图片
#     plt.show()
#     im = im.convert('L')
#     tv = list(im.getdata())
#     tva = [(255 - x) * 1.0 / 255.0 for x in tv]
#     return tva


# result = imageprepare()
# result = batch_y[0]

x = tf.placeholder(tf.float32, [None, 784])


# 给定卷集步幅
def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# 定义一个dropout
def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# 定义两个卷积层， 一个全连接层作为网络输出
def conv_net(x_, weights, biases, dropout_):
    x_ = tf.reshape(x_, shape=[-1, 28, 28, 1])
    #     x = tf.reshape(x, shape = [-1, 32, 32, 1])

    conv1 = conv2d(x_, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)

    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)

    # 匹配全连接层的输入
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # forward
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # relu
    fc1 = tf.nn.relu(fc1)
    # dropout
    fc1 = tf.nn.dropout(fc1, dropout_)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


with tf.Session() as sess:
    # 加载计算图
    saver = tf.train.import_meta_graph('./model.ckpt.meta')
    # 运行计算图
    sess.run(tf.global_variables_initializer())

    graph = tf.get_default_graph()
    node_list = graph.get_operations()

    # for i in node_list:
    #     print(i)

    # 权重
    w = {
        # 5x5 conv, 1 input, 32 output
        'wc1': graph.get_tensor_by_name('wc1:0'),
        # 5x5 conv, 32 input, 64 output
        'wc2': graph.get_tensor_by_name('wc2:0'),
        # 全连接层
        'wd1': graph.get_tensor_by_name('wd1:0'),
        # 输出层 1024input， n类输出
        'out': graph.get_tensor_by_name('out_w:0')
    }

    print(sess.run(w['wc1']))
    # biases
    b = {
        'bc1': graph.get_tensor_by_name('bc1:0'),
        'bc2': graph.get_tensor_by_name('bc2:0'),
        'bd1': graph.get_tensor_by_name('bd1:0'),
        'out': graph.get_tensor_by_name('out_b:0')
    }
    image = batch_x[0]

    # plt.imshow(image.reshape(28, 28))
    # plt.show()

    y_pre = conv_net(image, w, b, 1)

    print(y_pre.eval())
    prediction = tf.argmax(y_pre, axis=1)

    print("真实结果：{}".format(np.argmax(batch_y[0])))

    print('识别结果:')
    print(prediction.eval())
