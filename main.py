import re
import os
import copy
import numpy as np
import bert_ner_predict
from openpyxl import Workbook
from struction import Struction


class InputExample(object):
    def __init__(self, text=None, label=None):
        self.text = text
        self.label = label


def text_predict(text):
    '''
    句子级别的NER
    :param text: 输入为一条文本
    :return:
    '''
    rep_text = text
    tokens = re.split('[,，;；:：。]', rep_text)  # 按标点符号划分为tokens

    features = [[], []]  # 第一个保存tokens对应的input_ids, 第二个保存tokens对应的input_mask
    for token in tokens:  # 对于每一个tokens
        if token:
            example = InputExample(text=' '.join([i for i in token]), label=None)  # 实例化为类example
            feature = bert_ner_predict.convert_single_example(example)  # 将example转换为feature
            features[0].append(feature.input_ids)  # 保存input_ids
            features[1].append(feature.input_mask)  # 保存input_mask
    input_fn_dict = {'input_ids': features[0], 'input_mask': features[1]}
    result, label2id = bert_ner_predict.run_predict(input_fn_dict, is_file=False)  # NER预测，设置is_file=False
    predict_ids = list(result)
    entity, tags, _ = get_entity(predict_ids, tokens, label2id)  # 提取实体识别结果
    for i in range(len(entity)):
        print(entity[i], tags[i])
    print('----------------------------------------------------------')
    words_tags = []  # 一个样本的词语和标签
    for index in range(len(entity)):
        words_tags.append([entity[index], tags[index]])

    result = Struction(words_tags).struction()
    for rst in result:
        print(rst)


def batch_process(text_dir, xlsx_dir):
    '''
    :param text_dir: txt文本路径
    所有生成的文件均与txt文本在同一文件夹下
    :return:
    '''

    path, filename = os.path.split(text_dir)
    pred_tf_file = os.path.join(path, filename.split('.')[0]+'.tf_record')
    print('生成的tf_record测试文件路径：' + pred_tf_file)
    examples = []
    origin_words = []  # 保存原始tokens
    n_tokens = []  # 第i个元素表示第i个样本的有效tokens数

    with open(text_dir, 'r', encoding='utf8') as f:
        lines = f.readlines()

    start = int(0.8*len(lines))
    for line in lines[start:]:  # 一行代表一份描述
        if line.strip():
            rep_text = line.strip()
            tokens = re.split('[,，;；:：。]', rep_text)
            n_ids = 0
            for token in tokens:
                if token:
                    n_ids += 1
                    origin_words.append(token)
                    examples.append(InputExample(text=' '.join([i for i in token]), label=None))
            n_tokens.append(n_ids)
    bert_ner_predict.convert_examples2tf_record(examples, pred_tf_file)
    result, label2id = bert_ner_predict.run_predict(pred_tf_file)
    predict_ids = list(result)

    wb = Workbook()
    sheet = wb.active
    head_info = ['原始MRI文本', 'L(方位词)', 'P(主实体)', 'L(方位词)', 'S(附属实体)', 'A(属性)', 'V(值)']
    for letter_code in range(65, 72):
        sheet[chr(letter_code) + '1'] = head_info[letter_code - 65]
    n_raw = start
    list_len = 2

    start_tokens, end_tokens, map_i = 0, 0, 0
    entities, tags, locations = [], [], []
    for n in n_tokens:
        end_tokens += n
        temp_entity, temp_tags, temp_location = get_entity(predict_ids[start_tokens:end_tokens], origin_words[start_tokens:end_tokens], label2id)  # 提取一个样本实体识别结果
        start_tokens = copy.deepcopy(end_tokens)
        entities.append(temp_entity)
        tags.append(temp_tags)
        locations.append(temp_location)

        wts = []
        for index in range(len(temp_entity)):
            wts.append([temp_entity[index], temp_tags[index]])
        result = Struction(wts).struction()  # 调用结构化类将文本结构化
        raw_text = lines[n_raw]

        # <合并单元格写入原始MRI文本>
        sheet.merge_cells('A' + str(list_len) + ':' + 'A' + str(len(result) + list_len - 1))
        sheet['A' + str(list_len)] = raw_text
        n_raw += 1
        # </合并单元格写入原始MRI文本>

        # <写入结构化表>
        for row in range(len(result)):
            for code in range(66, 72):
                sheet[chr(code) + str(row + list_len)] = result[row][code - 66]
        list_len += len(result)
        # </写入结构化表>

    wb.save(xlsx_dir)  # 结构化结果保存到表格中


def get_entity(x_seq, tokens, label2id):
    '''
    获取NER实体，由write_result2file()调用
    :param x_seq: 预测序列
    :param tokens: 文字token，与x_seq一一对应
    :param label2id:
    :return:
    '''
    # 获取label2id中各实体类别BIS分别对应的id
    begin, inner, single, other, entity_name = [], [], [], [], []
    for key, value in label2id.items():
        if key.startswith('B'):
            begin.append(value)  # 获取各实体B的id
        elif key.startswith('I'):
            inner.append(value)  # 获取各实体I的id
        elif key.startswith('S'):
            single.append(value)  # 获取各实体S的id
            entity_name.append(re.split('-', key)[-1])  # 同时获取实体名
        elif key.startswith('O'):
            other.append(value)
        else:
            continue

    tag_series = ''.join(entity_name)
    entity = []
    tags = []
    location = []
    loc_index = 0
    for index in range(len(x_seq)):
        sample_xs = x_seq[index]  # 取第index个token的id
        words_xs = tokens[index]  # 取第index个token
        entity_xs = []
        tag_xs = []
        loc_xs = []
        i = 0

        while i < (len(sample_xs)):
            if sample_xs[i] in begin:  # 第i个元素为begin
                start_index = i  # 实体开始索引
                end_index = 0  # 初始化结束索引为0
                for j in range(i+1, len(sample_xs)):
                    if sample_xs[j] == sample_xs[start_index] + 1:  # 后续元素值等于开始索引元素值加1
                        end_index = j  # 实体结束索引
                    else:
                        break
                i = j
                if end_index:  # 如果结束索引不为0
                    entity_xs.append(words_xs[start_index:end_index+1])  # 获取实体
                    tag_xs.append(tag_series[begin.index(sample_xs[start_index])])  # 获取实体标签
                    loc_xs.append((start_index + end_index)/2 + loc_index)  # 获取实体中心位置
            elif sample_xs[i] in single:  # 第i个元素为single
                entity_xs.append(words_xs[i])  # 获取实体
                tag_xs.append(tag_series[single.index(sample_xs[i])])
                loc_xs.append(i + loc_index)
                i += 1
            elif sample_xs[i] in other:
                tago = words_xs[i]
                while i+1 <= len(sample_xs):
                    if sample_xs[i+1] in other:
                        tago += words_xs[i+1]
                        i += 1
                    else:
                        break
                entity_xs.append(tago)  # 获取实体
                tag_xs.append('O')
                loc_xs.append(0)
                i += 1
            else:
                i += 1
        if entity_xs:
            entity.append(entity_xs)
            tags.append(tag_xs)
            location.append(loc_xs)
        loc_index += len(np.nonzero(sample_xs)[0])
    return entity, tags, location


if __name__ == '__main__':
    # texta = '左侧咽旁间隙多发肿大淋巴结，大小2.6×1.5cm。'
    # textb = '左侧咽旁间隙多发大小为2.6×1.5cm的淋巴结。'
    # text_predict(texta)
    # text_predict(textb)

    batch_process('data/naso_mri_reports.txt', 'data/naso_struction.xlsx')

