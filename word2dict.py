import re
import random
import pickle


def split_corpus():
    '''
    划分语料为训练集(train)、验证集(eval)、测试集(test)，比例为6：2：2
    :param raw_txt:
    :return:
    '''
    with open('data/naso_ct_reports.txt', 'r', encoding='utf8') as f_raw:  # 打开要生成NER语料的文本数据
        raw_lines = f_raw.readlines()

    num_of_samples = len(raw_lines)

    train_len = int(0.6*num_of_samples)
    eval_len = int(0.2*num_of_samples)
    # test_len = num_of_samples - train_len - eval_len

    rand_index = random.sample([i for i in range(num_of_samples)], num_of_samples)  # 将原始文本数据的索引重新排列

    train_index = rand_index[:train_len]  # 按索引获取训练集
    dev_index = rand_index[train_len:train_len+eval_len]  # 按索引获取验证集
    test_index = rand_index[train_len+eval_len:num_of_samples]  # 按索引获取测试集

    f_raw_train = open('data/raw_train_reports.txt', 'w', encoding='utf8')  # 写入训练集
    for ti in train_index:
        f_raw_train.write(raw_lines[ti])
    f_raw_train.close()

    f_raw_dev = open('data/raw_dev_reports.txt', 'w', encoding='utf8')  # 写入验证集
    for ti in dev_index:
        f_raw_dev.write(raw_lines[ti])
    f_raw_dev.close()

    f_raw_test = open('data/raw_test_reports.txt', 'w', encoding='utf8')  # 写入测试集
    for ti in test_index:
        f_raw_test.write(raw_lines[ti])
    f_raw_test.close()


def id_map():
    '''
    生成文字到id，标签到id的映射表，并保存在'data/maps.pkl'中
    '''
    with open('data/naso_train.txt', 'r', encoding='utf8') as f:
        lines = f.readlines()
    with open('data/naso_test.txt', 'r', encoding='utf8') as f1:
        lines1 = f1.readlines()
    with open('data/naso_dev.txt', 'r', encoding='utf8') as f2:
        lines2 = f2.readlines()

    words = []
    for line in lines:
        if line.strip():
            word_tag = re.split(' ', line.strip())
            words.append(word_tag[0])
    for line in lines1:
        if line.strip():
            word_tag = re.split(' ', line.strip())
            words.append(word_tag[0])
    for line in lines2:
        if line.strip():
            word_tag = re.split(' ', line.strip())
            words.append(word_tag[0])

    words = set(words)
    word_nums = [i for i in range(1, len(words)+1)]
    id2word = dict(zip(word_nums, words))
    id2word[0] = ''
    word2id = dict(zip(words, word_nums))
    word2id[''] = 0

    tag_nums = [i for i in range(0, 21)]
    tags = ['']
    str1 = 'LPSAV'
    str2 = 'BIS'
    for i in str1:
        for j in str2:
            tags.append(j + '-' + i)
    tags += ['S-C', 'O']
    id2tag = dict(zip(tag_nums, tags))
    tag2id = dict(zip(tags, tag_nums))
    with open('data/maps.pkl', 'wb') as f:
        pickle.dump(word2id, f)  # 保存字映射到id
        pickle.dump(tag2id, f)  # 保存标签映射到id
        pickle.dump(id2word, f)  # 保存id映射到字
        pickle.dump(id2tag, f)  # 保存id映射到标签
    print('映射字典已保存')
    print('标签示例：', tag2id)


def text2id(txt_dir, pkl_dir):
    '''
    将文本描述转换为id矩阵，并保存到.pkl文件中
    :param txt_dir: 文本文件的路径
    :param pkl_dir: .pkl文件的路径
    '''
    # load maps.pkl##########################################
    with open('data/maps.pkl', 'rb') as f:
        word2id = pickle.load(f)
        tag2id = pickle.load(f)
        # id2word = pickle.load(f)
        # id2tag = pickle.load(f)
    #########################################################


    with open(txt_dir, 'r', encoding='utf8') as f:
        lines = f.readlines()
    words = []
    tags = []
    words_ids = []
    tags_ids = []
    sequence_lengths = []
    for line in lines:
        if line.strip():
            word_tag = re.split(' ', line.strip())
            words.append(word_tag[0])
            tags.append(word_tag[1])
        else:
            if len(words) != 0:
                words_ids.append(padding(words, word2id)[0])
                sequence_lengths.append(padding(words, word2id)[1])
                tags_ids.append(padding(tags, tag2id)[0])
            else:
                pass
            words, tags = [], []

    with open(pkl_dir, 'wb') as f_corpus:
        pickle.dump(words_ids, f_corpus)
        pickle.dump(tags_ids, f_corpus)
        pickle.dump(sequence_lengths, f_corpus)


def padding(sequence, xdict):
    max_seq = 50
    trans = []
    for key in sequence:
        trans.append(xdict[key])
    seq_len = len(trans)
    if seq_len <= max_seq:
        trans += [0]*(max_seq-seq_len)
        return trans, seq_len
    else:
        return trans[:max_seq], max_seq


if __name__ == '__main__':
    id_map()
    text2id('data/naso_train.txt', 'data/naso_id_train.pkl')
    text2id('data/naso_test.txt', 'data/naso_id_test.pkl')
    text2id('data/naso_dev.txt', 'data/naso_id_dev.pkl')
