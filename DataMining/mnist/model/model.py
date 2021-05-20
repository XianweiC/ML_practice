# -*- coding =utf-8 -*-
# @Project : DataMining
# @Time : 2021/5/10/13:44
# @Author : XianweiCao
# @Package :
# @File : model.py
# @Software: PyCharm Professional

import tensorflow as tf

learning_rate = 1e-2
training_iters = 500
batch_size = 500
display_step = 10

# 图片shape 32x32x3 = 3072
# n_input = 3072

n_input = 784
# 共10类
n_classes = 10

dropout = 0.85

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot='true')

x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)


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
def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    #     x = tf.reshape(x, shape = [-1, 32, 32, 1])

    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
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
    fc1 = tf.nn.dropout(fc1, dropout)

    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    # 保存模型输出
    # tf.add_to_collection('network-output', out)

    return out


# 权重
weights = {
    # 5x5 conv, 1 input, 32 output
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='wc1'),
    # 5x5 conv, 32 input, 64 output
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='wc2'),
    # 全连接层
    'wd1': tf.Variable(tf.random_normal([7 * 7 * 64, 1024]), name='wd1'),
    # 输出层 1024input， n类输出
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='out_w')
}

# biases
biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='out_b')
}

# model
pred = conv_net(x, weights, biases, keep_prob)
# 损失
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,
                                                              labels=y))
# 使用Adam优化器
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1),
                              tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 计算图初始化
init = tf.global_variables_initializer()

train_loss = []
train_acc = []
test_acc = []

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step <= training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # print(batch_x.shape)
        # print(batch_y.shape)
        sess.run(optimizer, feed_dict={x: batch_x,
                                       y: batch_y,
                                       keep_prob: 1.})
        if step % display_step == 0:
            loss_train, acc_train = sess.run([cost, accuracy],
                                             feed_dict={x: batch_x,
                                                        y: batch_y,
                                                        keep_prob: 1.})
            print("Iter" + str(step) + ", Minibatch Loss=" +
                  ":{}".format(loss_train) + ", Training Accuracy=" +
                  ":{}".format(acc_train))
            acc_test = sess.run(accuracy,
                                feed_dict={x: batch_x,
                                           y: batch_y,
                                           keep_prob: 1.})
            print("Testing Accuracy:" +
                  ":{}".format(acc_train))
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_acc.append(acc_test)
        step += 1
    saver = tf.train.Saver()
    model_path = './model.ckpt'
    save_path = saver.save(sess, model_path)

    #  保存计算图
    graph = tf.summary.merge_all()
    tf.summary.FileWriter('/summary', graph)

import matplotlib.pyplot as plt

eval_indices = range(0, training_iters, display_step)

plt.plot(eval_indices, train_loss, 'k-')
plt.title("Softmax Loss per iteration")
plt.xlabel('Iteration')
plt.ylabel('Softmax Loss')
plt.show()

plt.plot(eval_indices, train_acc, 'k-')
plt.plot(eval_indices, test_acc, 'r--')
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
