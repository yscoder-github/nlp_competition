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
    def __init__(self,embedding, batch_size=args.BATCH):
        '''
        :param scope_name:
        :param iterator: 调用tensorflow DataSet API把数据feed进来。
        :param embedding: 提前训练好的word embedding
        :param batch_size:
        '''
        self.batch_size = batch_size
        self.embedding = embedding
        # ——————————————————导入数据——————————————————————
        self.input= tf.placeholder(tf.int32, shape=[None, None],name="input")
        self.label = tf.placeholder(tf.int32, shape=[None, None],name="label")
        self.seq_length = tf.placeholder(tf.int32, shape=[None], name="max_sequence_in_batch")
        self._build_net()

    
        def multihead_attn(queries, keys, q_masks, k_masks, num_units=None, num_heads=8,
                        dropout_rate=DROPOUT_RATE, future_binding=False, reuse=False, activation=None):
            """
            Args:
            queries: A 3d tensor with shape of [N, T_q, C_q]
            keys: A 3d tensor with shape of [N, T_k, C_k]
            """
            if num_units is None:
                num_units = queries.get_shape().as_list[-1]
            T_q = queries.get_shape().as_list()[1]  # max time length of query
            T_k = keys.get_shape().as_list()[1]  # max time length of key

            Q = tf.layers.dense(queries, num_units, activation, reuse=reuse, name='Q')  # (N, T_q, C)
            K_V = tf.layers.dense(keys, 2 * num_units, activation, reuse=reuse, name='K_V')
            K, V = tf.split(K_V, 2, -1)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

            # Scaled Dot-Product
            align = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)
            align = align / np.sqrt(K_.get_shape().as_list()[-1])  # scale

            # Key Masking
            paddings = tf.fill(tf.shape(align), float('-inf'))  # exp(-large) -> 0

            key_masks = k_masks  # (N, T_k)
            key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, T_q, 1])  # (h*N, T_q, T_k)
            align = tf.where(tf.equal(key_masks, 0), paddings, align)  # (h*N, T_q, T_k)

            if future_binding:
                lower_tri = tf.ones([T_q, T_k])  # (T_q, T_k)
                lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()  # (T_q, T_k)
                masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])  # (h*N, T_q, T_k)
                align = tf.where(tf.equal(masks, 0), paddings, align)  # (h*N, T_q, T_k)

            # Softmax
            align = tf.nn.softmax(align)  # (h*N, T_q, T_k)

            # Query Masking
            query_masks = tf.to_float(q_masks)  # (N, T_q)
            query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, T_k])  # (h*N, T_q, T_k)
            align *= query_masks  # (h*N, T_q, T_k)

            align = tf.layers.dropout(align, dropout_rate, training=(not reuse))  # (h*N, T_q, T_k)

            # Weighted sum
            outputs = tf.matmul(align, V_)  # (h*N, T_q, C/h)
            # Restore shape
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
            # Residual connection
            outputs += queries  # (N, T_q, C)
            # Normalize
            outputs = layer_norm(outputs)  # (N, T_q, C)
            return outputs

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
        self.train_op = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.loss)
        tf.summary.scalar('loss', self.loss)

#训练神经网络

embedding = load_word2vec_embedding(config.vocab_size)
net = NER_net(embedding)


with tf.Session() as sess:
    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    writer = tf.summary.FileWriter(LOG_PATH, sess.graph)  # 将训练日志写入到logs文件夹下
    sess.run(tf.global_variables_initializer())
    print(dataset.get_step())
    for i in range(dataset.get_step()):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)

        max_sentenc_length = max(map(len, x_train))
        sequence_len = np.asarray([len(x) for x in x_train])
        # padding
        x_train = np.asarray([list(x[:]) + (max_sentenc_length - len(x)) * [config.src_padding] for x in x_train])
        y_train = np.asarray([list(y[:]) + (max_sentenc_length - len(y)) * [TAGS_NUM - 1] for y in y_train])
        res,loss_,_= sess.run([merged, net.loss, net.train_op], feed_dict={net.input: x_train, net.label: y_train, net.seq_length: sequence_len})
        print('steps:{}loss:{}'.format(i, loss_))
        writer.add_summary(res, i)  # write log into file 
        if i % 50 == 0:
            modelpp.save_model(sess, MODEL_PATH, overwrite=True)

