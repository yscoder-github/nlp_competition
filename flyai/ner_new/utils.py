import json
import numpy as np
import config
import tensorflow as tf




def load_word2vec_embedding(vocab_size):
    '''
        加载外接的词向量。
        :return:
    '''
    print('loading word embedding, it will take few minutes...')
    embeddings = np.random.uniform(-1,1,(vocab_size + 2, config.embeddings_size))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(23455)
    unknown = np.asarray(rng.normal(size=(config.embeddings_size)))
    padding = np.asarray(rng.normal(size=(config.embeddings_size)))
    f = open(config.word_embedding_file)
    embedding_table = json.load(f)

    with open(config.src_vocab_file,'r') as fw:
        words_dic=json.load(fw)
    for index, line in enumerate(embedding_table):
        values = embedding_table[line]
        try:
            coefs = np.asarray(values, dtype='float32')  # 取向量
        except ValueError:
            # 如果真的这个词出现在了训练数据里，这么做就会有潜在的bug。那coefs的值就是上一轮的值。
            print(line, values)
        if line in words_dic:
            embeddings[int(words_dic[line])] = coefs   # 将词和对应的向量存到字典里
    f.close()
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, config.embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)