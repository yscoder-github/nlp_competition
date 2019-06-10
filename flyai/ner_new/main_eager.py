# -*- coding: utf-8 -*
import argparse
from flyai.dataset import Dataset
from tensorflow.contrib.rnn import DropoutWrapper
import tensorflow as tf
from model import Model
from path import MODEL_PATH, LOG_PATH
import config
from utils import load_word2vec_embedding
import numpy as np
tf.enable_eager_execution()

# 超参 
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=30, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)



# 模型操作辅助类
modelpp = Model(dataset)


'''
使用tensorflow实现自己的算法

'''
# 得到训练和测试的数据
unit_num = config.embeddings_size      # 默认词向量的大小等于RNN(每个time step) 和 CNN(列) 中神经单元的个数, 为了避免混淆model中全部用unit_num表示。
time_step = config.max_sequence      # 每个句子的最大长度和time_step一样,为了避免混淆model中全部用time_step表示。
DROPOUT_RATE = config.dropout
LEARN_RATE=config.leanrate
TAGS_NUM = config.label_len

# ——————————————————定义神经网络变量——————————————————
class NER_net:
    def __init__(self,embedding, input, label, seq_length, batch_size=args.BATCH):
        '''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.embedding = embedding
        # ——————————————————导入数据——————————————————————
        # self.input= tf.placeholder(tf.int32, shape=[None, None],name="input")
        # self.label = tf.placeholder(tf.int32, shape=[None, None],name="label")
        # self.seq_length = tf.placeholder(tf.int32, shape=[None], name="max_sequence_in_batch")
        self.input = input
        self.label = label
        self.seq_length = sequence_len
        self._build_net()

    def _build_net(self):

        # x: [batch_size, time_step, embedding_size], float32
        self.x = tf.nn.embedding_lookup(self.embedding, self.input)
        # y: [batch_size, time_step]
        self.y = self.label

        cell_forward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        cell_backward = tf.contrib.rnn.BasicLSTMCell(unit_num)
        if DROPOUT_RATE is not None:
            cell_forward = DropoutWrapper(cell_forward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)
            cell_backward = DropoutWrapper(cell_backward, input_keep_prob=1.0, output_keep_prob=DROPOUT_RATE)

        # time_major 可以适应输入维度。
        outputs, bi_state = \
            tf.nn.bidirectional_dynamic_rnn(cell_forward, cell_backward, self.x, dtype=tf.float32)

        forward_out, backward_out = outputs
        outputs = tf.concat([forward_out, backward_out], axis=2)

        # projection:
        W = tf.get_variable("projection_w", [2 * unit_num, TAGS_NUM])
        b = tf.get_variable("projection_b", [TAGS_NUM])
        x_reshape = tf.reshape(outputs, [-1, 2 * unit_num])
        projection = tf.add(tf.matmul(x_reshape, W), b, name='projection')
        nsteps = tf.shape(outputs)[1]
        # -1 to time step
        self.outputs = tf.reshape(projection, [-1, nsteps, TAGS_NUM], name='output')

        self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(
            self.outputs, self.y, self.seq_length)
        self.transition_params = tf.add(self.transition_params, 0, name='transition_params')
        # Add a training op to tune the parameters.
        self.loss = tf.reduce_mean(-self.log_likelihood)
        tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)

#训练神经网络


embedding = load_word2vec_embedding(config.vocab_size)



for i in range(dataset.get_step()):
    x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)

    max_sentenc_length = max(map(len, x_train))
    sequence_len = np.asarray([len(x) for x in x_train])
    # padding
    x_train = np.asarray([list(x[:]) + (max_sentenc_length - len(x)) * [config.src_padding] for x in x_train], dtype=np.int32)
    y_train = np.asarray([list(y[:]) + (max_sentenc_length - len(y)) * [TAGS_NUM - 1] for y in y_train], dtype=np.int32)

    x_train = tf.Variable(x_train, dtype=tf.int32)
    y_train = tf.Variable(y_train, dtype=tf.int32)
    net = NER_net(embedding, x_train, y_train, sequence_len )

    # res,loss_,_= sess.run([merged, net.loss, net.train_op], feed_dict={net.input: x_train, net.label: y_train, net.seq_length: sequence_len})
    print('steps:{}loss:{}'.format(i, net.loss_))
    # writer.add_summary(res, i)  # write log into file 
    if i % 50 == 0:
        modelpp.save_model(sess, MODEL_PATH, overwrite=True)





