# coding=utf-8

import tensorflow as tf
from bert import tokenization, modeling
import os
from  bert.run_classifier import convert_single_example_simple

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def get_inputdata(query):
    token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
    split_tokens = token.tokenize(query)
    word_ids = token.convert_tokens_to_ids(split_tokens)
    word_mask= [1] * len(word_ids)
    word_segment_ids = [0] * len(word_ids)
    return word_ids,word_mask,word_segment_ids

# 配置文件
data_root = './bert/weight/chinese_L-12_H-768_A-12/'
bert_config_file = data_root + 'bert_config.json'
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
init_checkpoint = data_root + 'bert_model.ckpt'
bert_vocab_file = data_root + 'vocab.txt'
bert_vocab_En_file = './bert/weight/uncased_L-12_H-768_A-12/vocab.txt'

# graph
input_ids = tf.placeholder(tf.int32, shape=[None, None], name='input_ids')
input_mask = tf.placeholder(tf.int32, shape=[None, None], name='input_masks')
segment_ids = tf.placeholder(tf.int32, shape=[None, None], name='segment_ids')

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
output_layer = model.get_sequence_output()# 这个获取每个token的output 输出[batch_size, seq_length, embedding_size] 如果做seq2seq 或者ner 用这个
output_layer_pooled = model.get_pooled_output() # 这个获取句子的output

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    query = u'今天去哪里吃'
    # word_ids, word_mask, word_segment_ids=get_inputdata(query)
    token = tokenization.CharTokenizer(vocab_file=bert_vocab_file)
    word_ids, word_mask, word_segment_ids=convert_single_example_simple(max_seq_length=32,tokenizer=token,text_a=query,text_b='这里吃')
    print(len(word_ids))
    print(word_mask)
    print(word_segment_ids)
    fd = {input_ids: [word_ids], input_mask: [word_mask], segment_ids: [word_segment_ids]}
    last, last2 = sess.run([output_layer, output_layer_pooled], feed_dict=fd)
    print('last shape:{}, last2 shape: {}'.format(last.shape, last2.shape))
    pass



