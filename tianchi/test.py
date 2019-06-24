import numpy as np
import wordfreq
vocab = {}
token_id = 1
lengths = []

f = [
     'To the world you may be one person, but to one person you may be the world.',
     'Never frown, even when you are sad, because you never know who is falling in love with your smile. ',\
     'We met at the wrong time, but separated at the right time. The most urgent is to take the most beautiful scenery; the deepest wound was the most real emotions.'
]
for l in f:
    tokens = wordfreq.tokenize(l.strip(), 'en')
    lengths.append(len(tokens))
    for t in tokens:
        if t not in vocab:
            vocab[t] = token_id
            token_id += 1

x = np.zeros((len(lengths), max(lengths)))
l_no = 0
with open('test.txt', 'r') as f:
    for l in f:
        tokens = wordfreq.tokenize(l.strip(), 'en')
        for i in range(len(tokens)):
            x[l_no, i] = vocab[tokens[i]]
        l_no += 1

