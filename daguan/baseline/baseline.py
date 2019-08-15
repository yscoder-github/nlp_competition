import codecs
import os

# 0 install crf++ https://taku910.github.io/crfpp/
# 1 train data in
# 2 test data in
# 3 crf train
# 4 crf test
# 5 submit test

# step 1 train data in
with codecs.open('train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    results = []
    for line in lines:
        features = []
        tags = []
        samples = line.strip().split('  ')
        for sample in samples:
            sample_list = sample[:-2].split('_')
            tag = sample[-1]
            features.extend(sample_list)
            tags.extend(['O'] * len(sample_list)) if tag == 'o' else tags.extend(['B-' + tag] + ['I-' + tag] * (len(sample_list)-1))
        results.append(dict({'features': features, 'tags': tags}))
    train_write_list = []
    with codecs.open('dg_train.txt', 'w', encoding='utf-8') as f_out:
        for result in results:
            for i in range(len(result['tags'])):
                train_write_list.append(result['features'][i] + '\t' + result['tags'][i] + '\n')
            train_write_list.append('\n')
        f_out.writelines(train_write_list)

# step 2 test data in
with codecs.open('test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    results = []
    for line in lines:
        features = []
        sample_list = line.split('_')
        features.extend(sample_list)
        results.append(dict({'features': features}))
    test_write_list = []
    with codecs.open('dg_test.txt', 'w', encoding='utf-8') as f_out:
        for result in results:
            for i in range(len(result['features'])):
                test_write_list.append(result['features'][i] + '\n')
            test_write_list.append('\n')
        f_out.writelines(test_write_list)

# 3 crf train
crf_train = "crf_learn -f 3 template.txt dg_train.txt dg_model"
os.system(crf_train)

# 4 crf test
crf_test = "crf_test -m dg_model dg_test.txt -o dg_result.txt"
os.system(crf_test)

# 5 submit data
f_write = codecs.open('dg_submit.txt', 'w', encoding='utf-8') 
with codecs.open('dg_result.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n\n')
    for line in lines:
        if line == '':
            continue
        tokens = line.split('\n')
        features = []
        tags = []
        for token in tokens:
            feature_tag = token.split()
            features.append(feature_tag[0])
            tags.append(feature_tag[-1])
        samples = []
        i = 0
        while i < len(features):
            sample = []
            if tags[i] == 'O':
                sample.append(features[i])
                j = i + 1
                while j < len(features) and tags[j] == 'O':
                    sample.append(features[j])
                    j += 1
                samples.append('_'.join(sample) + '/o')
            else:
                if tags[i][0] != 'B':
                    print(tags[i][0] + ' error start')
                    j = i + 1
                else:
                    sample.append(features[i])
                    j = i + 1
                    while j < len(features) and tags[j][0] == 'I' and tags[j][-1] == tags[i][-1]:
                        sample.append(features[j])
                        j += 1
                    samples.append('_'.join(sample) + '/' + tags[i][-1])
            i = j
        f_write.write('  '.join(samples) + '\n')
