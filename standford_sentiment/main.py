# -*- coding: utf-8 -*
import argparse
import tensorflow as tf
from flyai.dataset import Dataset
from model import Model
from path import MODEL_PATH
import tensorflow as tf
import bert.modeling as modeling
import os



# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# 模型操作辅助类
modelpp = Model(dataset)

path = modelpp.get_remote_date("https://www.flyai.com/m/multi_cased_L-12_H-768_A-12.zip")
'''
使用tensorflow实现自己的算法

'''
# 得到训练和测试的数据

lr = 0.0006  # 学习率
# 配置文件

data_root = os.path.splitext(path)[0]
bert_config_file = os.path.join(data_root, 'bert_config.json')
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = os.path.join(data_root, 'bert_model.ckpt')
bert_vocab_file = os.path.join(data_root, 'vocab.txt')
# ——————————————————导入数据——————————————————————

input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')
input_y = tf.placeholder(tf.float32, shape=[None, 1], name="input_y")
# ——————————————————定义神经网络变量——————————————————
# 输入层、输出层权重、偏置

#with tf.variable_scope('in_out',reuse=tf.AUTO_REUSE):
weights = {
    'out': tf.Variable(tf.random_normal([768, 1]))
}
biases = {
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}



# ——————————————————定义神经网络变量——————————————————
# 初始化BERT
model = modeling.BertModel(
    config=bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=False)

# 加载bert模型
tvars = tf.trainable_variables()
(assignment, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
tf.train.init_from_checkpoint(init_checkpoint, assignment)
# 获取最后一层。
#output_layer = model.get_sequence_output()# 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
output_layer_pooled = model.get_pooled_output() # 这个获取句子的output


# ——————————————————训练模型——————————————————

w_out = weights['out']
b_out = biases['out']
pred = tf.add(tf.matmul(output_layer_pooled, w_out), b_out, name="pre1")
pred=tf.reshape(pred,shape=[-1,1],name="pre")


# # 损失函数
loss=tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(input_y, [-1])))
tf.summary.scalar('loss',loss)
#
train_op = tf.train.AdamOptimizer(lr).minimize(loss)
with tf.Session() as sess:
    merged = tf.summary.merge_all()  # 将图形、训练过程等数据合并在一起
    writer = tf.summary.FileWriter('./logs', sess.graph)  # 将训练日志写入到logs文件夹下
    sess.run(tf.global_variables_initializer())
    for i in range(dataset.get_step()):
        x_train, y_train, x_test, y_test = dataset.next_batch(args.BATCH)
        x_input_ids=x_train[0]
        x_input_mask=x_train[1]
        x_segment_ids=x_train[2]
#        sess.run(train_op, feed_dict={input_ids: x_input_ids,input_mask: x_input_mask,segment_ids: x_segment_ids, input_y: y_train})
        res,loss_,_= sess.run([merged,loss,train_op],feed_dict={input_ids: x_input_ids,input_mask: x_input_mask,segment_ids: x_segment_ids, input_y: y_train})
        print('steps:{}loss:{}'.format(i,loss_))
        writer.add_summary(res, i)  # 将日志数据写入文件
        if i%1000==0:
            modelpp.save_model(sess, MODEL_PATH, overwrite=True)
