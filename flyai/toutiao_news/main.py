# -*- coding: utf-8 -*
import argparse
import tensorflow as tf
from flyai.dataset import Dataset

from model import Model
from path import MODEL_PATH

# 数据获取辅助类
dataset = Dataset()
# 模型操作辅助类
model = Model(dataset)

# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=20, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
max_seq_len = 50
'''
使用tensorflow实现自己的算法

'''

# 得到训练和测试的数据
# batch_size=100
time_step = 5  # lstm序列长度
input_size = 200

rnn_unit = 200  # hidden layer units

output_size = 15
lr = 0.0006  # 学习率
# ——————————————————导入数据——————————————————————

input_x = tf.placeholder(tf.float32, shape=[None, max_seq_len, input_size], name="input_x")
input_y = tf.placeholder(tf.float32, shape=[None, output_size], name="input_y")
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

# with tf.variable_scope('in_out',reuse=tf.AUTO_REUSE):
weights = {
    'out': tf.Variable(tf.random_normal([2 * max_seq_len * rnn_unit, 15]))
}
biases = {
    'out': tf.Variable(tf.constant(0.1, shape=[15, ]))
}


# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    #    with tf.variable_scope('lstm',reuse=tf.AUTO_REUSE):
    input_rnn = tf.reshape(X, [-1, max_seq_len, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    s = tf.shape(input_rnn)
    # bi lstm on words
    cell_fw = tf.contrib.rnn.LSTMCell(rnn_unit,
                                      state_is_tuple=True)
    cell_bw = tf.contrib.rnn.LSTMCell(rnn_unit,
                                      state_is_tuple=True)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, input_rnn,
        dtype=tf.float32)
    output_fw, output_bw = outputs
    states_fw, states_bw = states
    # read and concat output
    output_word = tf.concat([output_fw, output_bw], axis=-1)

    # shape = (batch size, max sentence length, char hidden size)
    output_w = tf.reshape(output_word,
                          shape=[s[0], s[1], 2 * rnn_unit])

    output = tf.reshape(output_w, [-1, 2 * max_seq_len * rnn_unit])  # 作为输出层的输入
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.add(tf.matmul(output, w_out), b_out, name="pre")
    return pred


# ——————————————————训练模型——————————————————

# with tf.variable_scope('pre',reuse=tf.AUTO_REUSE):
y = lstm(input_x)
y_conv = tf.argmax(y, name="y_conv", axis=1)

# # 损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=input_y, name=None))
#
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(args.EPOCHS):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
        #        y = sess.run(y_conv, feed_dict={input_x: x_train})
        sess.run(train_op, feed_dict={input_x: x_train, input_y: y_train})
        loss_ = sess.run(loss, feed_dict={input_x: x_train, input_y: y_train})
        print('loss:', loss_)

    model.save_model(sess, MODEL_PATH, overwrite=True)
