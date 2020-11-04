import re
import os
import codecs
import numpy as np
from matplotlib import pyplot as plt


class Evaluator(object):
    def __init__(self, file_dir, label2id):
        self.file_dir = file_dir
        self.label2id = label2id
        self.begin = []
        self.inner = []
        self.single = []
        self.entity_name = []

        self.__entity_flags()
        self.__id_seq()

    def __entity_flags(self):
        '''
        called by self.__init__()
        '''
        for key, value in self.label2id.items():
            if key.startswith('B'):
                self.begin.append(value)
            elif key.startswith('I'):
                self.inner.append(value)
            elif key.startswith('S'):
                self.single.append(value)
                self.entity_name.append(re.split('-', key)[-1])
            else:
                continue

    def __id_seq(self):
        '''
        called by self.__init__()
        '''
        with codecs.open(self.file_dir, 'r', encoding='utf8') as f:
            self.lines = f.readlines()
        self.x_seq, self.y_seq = [], []
        x_tags, y_tags = [], []
        for line in self.lines:
            if line.strip():
                segments = line.strip().split(' ')
                if len(segments) == 3:
                    x_tags.append(self.label2id[segments[1]])  # 将实际的标签转换为id
                    y_tags.append(self.label2id[segments[2]])  # 将预测的标签转换为id
            else:
                self.x_seq.append(x_tags)
                self.y_seq.append(y_tags)
                x_tags, y_tags = [], []

    def all_entity_metrics(self, x_seq, y_seq):
        [total_entity, correct_entity] = [0, 0]
        for index in range(len(x_seq)):
            sample_xs = x_seq[index]  # 取第index个样本
            sample_ys = y_seq[index]
            i = 0

            while i < (len(sample_xs)):
                if sample_xs[i] in self.begin:  # 第i个元素为begin
                    start_index = i  # 实体开始索引
                    end_index = 0  # 初始化结束索引为0
                    if i == len(sample_xs) - 2:
                        j = len(sample_xs) - 1
                        if sample_xs[j] == sample_xs[start_index] + 1:  # 后续元素值等于开始索引元素值加1
                            end_index = j  # 实体结束索引
                        i = j
                    elif i == len(sample_xs) - 1:
                        i += 1
                    else:
                        for j in range(i + 1, len(sample_xs)):
                            if sample_xs[j] == sample_xs[start_index] + 1:  # 后续元素值等于开始索引元素值加1
                                end_index = j  # 实体结束索引
                            else:
                                break
                        i = j

                    if end_index:  # 如果结束索引不为0
                        xs_entity = sample_xs[start_index:end_index + 1]  # 获取实体
                        ys_entity = sample_ys[start_index:end_index + 1]
                        total_entity += 1  # 总实体数加1
                        # print(xs_entity, ys_entity)  #################################################################
                        if xs_entity == ys_entity:  # 如果相等，正确实体数加1
                            correct_entity += 1

                elif sample_xs[i] in self.single:  # 第i个元素为single
                    xs_entity = sample_xs[i]  # 获取实体
                    ys_entity = sample_ys[i]
                    total_entity += 1  # 总实体数加1
                    # print(xs_entity, ys_entity)  #####################################################################
                    if xs_entity == ys_entity:  # 如果相等，正确实体数加1
                        correct_entity += 1

                    i += 1
                else:
                    i += 1
        return correct_entity / (total_entity + 10e-5)

    def precise_entity_metrics(self, x_seq, y_seq):
        n_entity = len(self.entity_name)
        total_entity = [0] * n_entity
        correct_entity = [0] * n_entity
        # [total_entity, correct_entity] = [0, 0]
        for index in range(len(x_seq)):
            sample_xs = x_seq[index]  # 取第index个样本
            sample_ys = y_seq[index]
            i = 0

            while i < (len(sample_xs)):
                if sample_xs[i] in self.begin:  # 第i个元素为begin
                    start_index = i  # 实体开始索引
                    end_index = 0  # 初始化结束索引为0
                    if i == len(sample_xs) - 2:
                        j = len(sample_xs) - 1
                        if sample_xs[j] == sample_xs[start_index] + 1:  # 后续元素值等于开始索引元素值加1
                            end_index = j  # 实体结束索引
                        i = j
                    elif i == len(sample_xs) - 1:
                        i += 1
                    else:
                        for j in range(i + 1, len(sample_xs)):
                            if sample_xs[j] == sample_xs[start_index] + 1:  # 后续元素值等于开始索引元素值加1
                                end_index = j  # 实体结束索引
                            else:
                                break
                        i = j

                    if end_index:  # 如果结束索引不为0
                        xs_entity = sample_xs[start_index:end_index + 1]  # 获取实体
                        ys_entity = sample_ys[start_index:end_index + 1]
                        total_entity[self.begin.index(sample_xs[start_index])] += 1
                        # total_entity += 1  # 总实体数加1
                        # print(xs_entity, ys_entity)  #################################################################
                        if xs_entity == ys_entity:  # 如果相等，正确实体数加1
                            correct_entity[self.begin.index(sample_xs[start_index])] += 1
                            # correct_entity += 1

                elif sample_xs[i] in self.single:  # 第i个元素为single
                    xs_entity = sample_xs[i]  # 获取实体
                    ys_entity = sample_ys[i]
                    total_entity[self.single.index(sample_xs[i])] += 1
                    # total_entity += 1  # 总实体数加1
                    # print(xs_entity, ys_entity)  #####################################################################
                    if xs_entity == ys_entity:  # 如果相等，正确实体数加1
                        correct_entity[self.single.index(sample_xs[i])] += 1
                        # correct_entity += 1

                    i += 1
                else:
                    i += 1

        return np.array(correct_entity) / (np.array(total_entity) + 10e-5)

    def get_metrics(self, mode='all'):
        if mode is 'all':
            accuracy = self.all_entity_metrics(self.x_seq, self.y_seq)
            recall = self.all_entity_metrics(self.y_seq, self.x_seq)
            f1 = 2*accuracy*recall / (accuracy+recall)
            return {'mode': mode, 'entity_name': None, 'accuracy': accuracy, 'recall': recall, 'f1': f1}
        elif mode == 'precise':
            precise_acc = self.precise_entity_metrics(self.x_seq, self.y_seq)
            precise_rec = self.precise_entity_metrics(self.y_seq, self.x_seq)
            precise_f1 = 2*precise_acc*precise_rec / (precise_acc+precise_rec)
            return {'mode': mode, 'entity_name': self.entity_name, 'accuracy': precise_acc,
                    'recall': precise_rec, 'f1': precise_f1}


def sparse_result(result):
    mode = result['mode']
    entity_name = result['entity_name']
    acc = result['accuracy']
    rec = result['recall']
    f1 = result['f1']
    write_info = []
    if mode == 'all':
        write_info.append('Accuracy: %.4f\tRecall: %.4f\tf1 value: %.4f\n' % (acc, rec, f1))
        print(''.join(write_info))
    elif mode == 'precise':
        for name, acc_i, rec_i, f1_i in zip(entity_name, acc, rec, f1):
            write_info.append('entity %3s: Accuracy: %.4f\tRecall: %.4f\tf1 value: %.4f\n' % (name, acc_i, rec_i, f1_i))
        print(''.join(write_info))
    return ''.join(write_info)


def plot_loss(train_loss, dev_loss, fig_save_dir, model_name):
    x = [i for i in range(len(train_loss))]
    plt.title(model_name)
    plt.plot(x, train_loss, label='train_loss')
    plt.plot(x, dev_loss, label='dev_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    fig_name = model_name + '.png'
    plt.savefig(os.path.join(fig_save_dir, fig_name))
