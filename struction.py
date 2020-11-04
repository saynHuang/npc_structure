import re


class Struction(object):
    def __init__(self, word_tag):
        self.__wts = word_tag
        self.__pre = ''  # 主实体
        self.__left = ''  # 左方位词
        self.__right = ''  # 右方位词
        self.__sec = ''  # 附属实体
        self.__att = ''  # 描述词
        self.__value = ''  # 属性词
        self.__content = []  # 结构化内容

    def struction(self):
        result = []  # 结构化结果
        for wt in self.__wts:
            self.__left = ''
            self.__right = ''
            self.__sparse(wt)
        for i, sample in enumerate(self.__content):
            if sample[-1]:  # 若V存在，则打印输出
                if sample[1] or sample[3]:  # 如果P或S存在
                    result.append(sample)  # 直接添加到结构化列表
                elif i != 0:
                    if re.search('淋巴结|.*影', self.__content[i - 1][4]):  # 上一级A有‘淋巴结’字样
                        self.__content[i][3] = self.__content[i - 1][4]  # 直接继承为S
                    elif re.search('粘膜|骨质|病灶|肿块', self.__content[i - 1][3]):  # 上一级S有‘粘膜’字样
                        self.__content[i][3] = self.__content[i - 1][3]  # 继承为S
                    else:  # 否则，继承上一级的P和S
                        self.__content[i][1] = self.__content[i - 1][1]
                        self.__content[i][3] = self.__content[i - 1][3]
                    if sample[1] or sample[3]:
                        result.append(self.__content[i])  # 添加到结构化列表
        return result

    def __sparse(self, wt):
        self.__words = wt[0]  # 分词后的词语
        self.__tag = wt[1]  # 词对应的标签

        self.__preprocess()  # 预处理

        if 'C' in self.__tag:  # 如果有连接词
            self.__sub_connect()  # 额外处理连接词的结构化
        else:
            result = self.__generate_content(self.__words, self.__tag)
            self.__content.append(result)

    def __preprocess(self):
        '''
        1、鼻咽 粘膜 形成 肿块 [PSVS] ==> 若含有其他S标签，则肿块(病灶)标签由S改为A
        2、肿块影/淋巴结 增强扫描 轻度强化 [AAV]  ==> 若含有其他A标签，则（*影/淋巴结）标签由A改为S
        3、形态 及 信号 未见 异常[ACAVV]，ACA合并为一个A ==> 形态、信号 未见 异常[AVV]
        4、合并VCV为一个V，或者VV为一个V（与3类似）
        '''

        # <pre rule 1>
        s2a_flag = 0
        for i, tag in enumerate(self.__tag):
            if tag == 'S' and self.__words[i] not in ['病灶', '肿块']:
                s2a_flag = 1
        for i, tag in enumerate(self.__tag):
            if tag == 'S' and self.__words[i] in ['病灶', '肿块'] and s2a_flag == 1:
                self.__tag[i] = 'A'
        # </pre rule 1>

        # <pre rule 2>
        a2s_flag = 0
        for i, tag in enumerate(self.__tag):
            if tag == 'A' and '影' not in self.__words[i] and '淋巴结' not in self.__words[i]:
                a2s_flag = 1
        for i, tag in enumerate(self.__tag):
            if tag == 'A' and ('影' in self.__words[i] or '淋巴结' in self.__words[i]) and a2s_flag == 1:
                self.__tag[i] = 'S'
        # </pre rule 2>

        # <pre rule 3>
        aca = re.search('A(CA)+', ''.join(self.__tag))  # 查找符合情况3的字符串
        if aca:
            start_span = aca.span()[0]
            end_span = aca.span()[1]
            for loop in range(start_span + 1, end_span):
                if (end_span - loop) % 2 == 0:
                    self.__words[start_span] += '、'
                else:
                    self.__words[start_span] += self.__words[start_span + 1]
                self.__words.pop(start_span + 1)
                self.__tag.pop(start_span + 1)

        # 若之后还有AA存在，则再合并AA
        aa = re.search('AA+', ''.join(self.__tag))  # 查找符合情况3的字符串
        if aa:
            start_span = aa.span()[0]
            end_span = aa.span()[1]
            for loop in range(start_span + 1, end_span):
                self.__words[start_span] += '、'
                self.__words[start_span] += self.__words[start_span + 1]
                self.__words.pop(start_span + 1)
                self.__tag.pop(start_span + 1)
        # </pre rule 3>

        # <pre rule 4>
        # 先合并VCV为V
        vcv = re.search('V([CO]V)+', ''.join(self.__tag))  # 查找符合情况4的字符串
        if vcv:
            start_span = vcv.span()[0]
            end_span = vcv.span()[1]
            for loop in range(start_span + 1, end_span):
                if (end_span - loop) % 2 == 0:  # 如果loop索引位于[CO]处
                    self.__words[start_span] += '、'  # 把C或O的实体替换为顿号‘、’
                else:
                    self.__words[start_span] += self.__words[start_span + 1]  # loop不在[CO]处，即位于实体A处，直接相加
                self.__words.pop(start_span + 1)
                self.__tag.pop(start_span + 1)

        # 若之后还有VV存在，则再合并VV
        vv = re.search('VV+', ''.join(self.__tag))  # 查找符合情况4的字符串
        if vv:
            start_span = vv.span()[0]
            end_span = vv.span()[1]
            for loop in range(start_span + 1, end_span):
                self.__words[start_span] += '、'
                self.__words[start_span] += self.__words[start_span + 1]
                self.__words.pop(start_span + 1)
                self.__tag.pop(start_span + 1)
        # </pre rule 4>

    def __find_index(self, words, tag, target):
        '''
        寻找标签所在的位置
        :param target: (str)要寻找的标签；(value)'LPSAVC'
        :return: 索引值和对应的词
        '''
        try:
            index = tag.index(target)  # 如果tag中有target标签
            value = words[index]  # 则获取其对应的词语
        except ValueError:
            index = -1  # 否则索引值设置为-1
            value = ''  # 词语为空
        return index, value

    def __sub_connect(self):
        tag_series = ''.join(self.__tag)  # 将标签列表合并为字符串
        split_tag = re.split('C', tag_series)  # 按连接词划分token标签
        split_word = []
        token_start = 0
        for sp_tag in split_tag:
            token_end = token_start + len(sp_tag)
            split_word.append(self.__words[token_start:token_end])  # 获取划分好的token词
            token_start = token_end + 1
        result = []
        for i in range(len(split_tag)):
            if split_word[i]:  # 如果token词不为空（即排除掉第一个标签为C的情况）
                result.append(self.__generate_content(split_word[i], split_tag[i]))  # 生成结构化列表
        if len(result) > 1:
            for res_i in range(len(result)-1, 0, -1):  # 逆序补SAV
                for index in range(3, 6):
                    if result[res_i - 1][index]:
                        pass
                    else:
                        result[res_i - 1][index] += result[res_i][index]
            for res_i in range(1, len(result)):  # 正序补P
                if result[res_i][1]:
                    pass
                else:
                    result[res_i][1] = result[res_i - 1][1]
            for content in result:
                self.__content.append(content)
        elif result:
            self.__content.append(result[0])

    def __generate_content(self, words, tag):
        words.append(' ')  # 添加一个空值，防止索引超出列表范围
        tag += 'O'
        sec = self.__find_index(words, tag, 'S')[1]  # 获取'S'副部位词
        att = self.__find_index(words, tag, 'A')[1]  # 获取'A'描述词
        value = self.__find_index(words, tag, 'V')[1]  # 获取'V'属性词
        left = ''
        right = ''
        p_index, pre = self.__find_index(words, tag, 'P')  # 获取主部位的索引及对应的词
        if p_index != -1:  # 如果P存在，找方位词
            if tag[p_index - 1] is 'L':  # 如果P前一个标签为L
                left = words[p_index - 1]  # 则获取左方位词
            if tag[p_index + 1] is 'L':  # 同理获取右方位词
                right = words[p_index + 1]
        else:  # P不存在，若有方位词，则默认为右方位词
            right = self.__find_index(words, tag, 'L')[1]  # 获取'L'属性词
        return [left, pre, right, sec, att, value]


