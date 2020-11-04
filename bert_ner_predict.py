import os
import pickle
import codecs
import collections
import tensorflow as tf
from tensorflow.contrib import crf
from bert import modeling, tokenization
from tensorflow.contrib.layers.python.layers import initializers


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, text_a=None, label=None):
        self.text_a = text_a
        self.label = label


# 定义模型参数
model_dir = 'ner_train/BERT_CRF_Model'
bert_dir = 'E://chinese_L-12_H-768_A-12'
bert_config_file = os.path.join(bert_dir, 'bert_config.json')
vocab_file = os.path.join(bert_dir, 'vocab.txt')
initial_ckpt = os.path.join(bert_dir, 'bert_model.ckpt')
max_seq_length = 100
batch_size = 80  # 此处batch_size为模型处理的token数，而不是样本数

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
bert_config = modeling.BertConfig.from_json_file(bert_config_file)
####################################################################


def run_predict(tf_record_dir, is_file=True):
    '''
    加载训练好的BERT-BiLSTM-CRF模型进行命名实体识别

    [文本级别的NER]：原始txt文件，一行代表一个样本句子，将句子按标点符号划分为tokens。然后经过bert转为对应的input_ids
                    和input_mask，在集中写入tf_record文件，之后将tf_record文件路径作为参数传入函数run_predict()；
    [句子级别的NER]：则可以将其转化为features字典，格式为features = {'input_ids': [], 'input_mask':[]}，然后传入函数
                    run_predict()的tf_record_dir，并设置is_file=False。
    :param tf_record_dir: 输入的数据，格式为tf_record或者是单独的一个句子
    :param is_file: 默认为True，当is_file=True时，tf_record_dir接受的是tf_record文件的路径；
                    当is_file=False时，tf_record_dir接受的是字典（dict），包含一个句子的input_ids和input_mask

    :return: 返回预测的标签和label2id字典
    '''

    # 配置session
    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    # 配置estimator的运行参数
    run_config = tf.estimator.RunConfig(model_dir=model_dir, session_config=session_config)

    # 加载label2id的词典，并作为一个返回结果
    with codecs.open(os.path.join(model_dir, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
    #     id2label = {value: key for key, value in label2id.items()}
    # id2label[0] = ''
    # print(label2id)
    num_labels = len(label2id) + 1

    # 配置一些额外的参数
    params = {'num_labels': num_labels, 'init_checkpoint': initial_ckpt, 'batch_size': batch_size, 'is_file': is_file}

    # 构造Estimator
    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)


    # 读取record数据，进行预测
    input_fn = file_based_input_fn(tf_record_dir, max_seq_length)
    result = estimator.predict(input_fn=input_fn)
    return result, label2id


def model_fn(features, labels, mode, params):
    num_labels = params['num_labels']
    init_checkpoint = params['init_checkpoint']

    input_ids = features['input_ids']
    input_mask = features['input_mask']

    print('details of input_ids', input_ids)
    print('details of input_mask', input_mask)

    # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
    total_loss, logits, trans, pred_ids = create_model(
        bert_config, is_training=None, input_ids=input_ids, input_mask=input_mask, segment_ids=None, labels=None,
        num_labels=num_labels, use_one_hot_embeddings=False, dropout_rate=1.0)

    tf_vars = tf.trainable_variables()
    # 加载BERT模型
    if init_checkpoint:
        (assignment_map, _) = modeling.get_assignment_map_from_checkpoint(tf_vars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_ids)
    else:
        exit('\n************************************************\n'
             '   此Estimator仅用于预测(tf.estimator.predict)\n'
             '   请检查Estimator的模式'
             '\n************************************************\n')

    return output_spec


def file_based_input_fn(tf_record_file, seq_length):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        is_file = params['is_file']
        if is_file:
            d = tf.data.TFRecordDataset(tf_record_file)

            d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                           batch_size=batch_size,
                                                           num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                           ))
            d = d.prefetch(buffer_size=4)
            return d
        else:
            # 如果tf_record_file为字典，则直接生成'tf.data.Dataset'格式的数据
            d = tf.data.Dataset.from_tensor_slices(tf_record_file)
            d = d.batch(batch_size)
            return d

    return input_fn


def convert_examples2tf_record(examples, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for example in examples:
        feature = convert_single_example(example)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def convert_single_example(example):
    '''
    将一个example转换为feature
    :param example:
    :return:
    '''
    # tokens_a = [s for s in text]
    tokens_a = tokenizer.tokenize(example.text)
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0:max_seq_length]
    tokens = []
    for index, token in enumerate(tokens_a):
        tokens.append(token)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 把字转换为id
    input_mask = [1] * len(input_ids)  # 同时获取mask序列

    while len(input_ids) < max_seq_length:  # padding值max_seq_length
        input_ids.append(0)
        input_mask.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length

    feature = InputFeatures(input_ids=input_ids, input_mask=input_mask)
    return feature


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings,
                 dropout_rate=1.0):
    """
    创建X模型
    :param bert_config: bert 配置
    :param is_training:
    :param input_ids: 数据的idx 表示
    :param input_mask:
    :param segment_ids:
    :param labels: 标签的idx 表示
    :param num_labels: 类别数量
    :param use_one_hot_embeddings:
    :return:
    """
    # 使用数据加载BertModel,获取对应的字embedding

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )
    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embedding = model.get_sequence_output()
    max_seq_length = embedding.shape[1].value
    embedding_dims = embedding.shape[-1].value
    # 获取序列的真实长度
    used = tf.sign(tf.abs(input_ids))
    lengths = tf.reduce_sum(used, reduction_indices=1)  # [batch_size] 大小的向量，包含了当前batch中的序列长度

    # Linear-CRF layer
    if is_training:
        # lstm input dropout rate i set 0.9 will get best score
        embedded_chars = tf.nn.dropout(embedding, dropout_rate)
    else:
        embedded_chars = embedding

    with tf.variable_scope("fully_connect"):
        W = tf.get_variable("fc_W", shape=[embedding_dims, num_labels],
                            dtype=tf.float32, initializer=initializers.xavier_initializer())

        b = tf.get_variable("fc_b", shape=[num_labels], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        output = tf.reshape(embedded_chars,
                            shape=[-1, embedding_dims])  # [batch_size, embedding_dims]
        predict = tf.nn.xw_plus_b(output, W, b)
        out = tf.reshape(predict, shape=[-1, max_seq_length, num_labels], name='out')

    # Linear-CRF
    with tf.variable_scope("crf_loss"):
        trans = tf.get_variable(
            "transitions",
            shape=[num_labels, num_labels],
            initializer=initializers.xavier_initializer())
        if labels is None:
            loss = None
        else:
            log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                inputs=out,
                tag_indices=labels,
                transition_params=trans,
                sequence_lengths=lengths)
            loss = tf.reduce_mean(-log_likelihood, name='cost')
    # CRF decode, pred_ids 是一条最大概率的标注路径
    pred_ids, _ = crf.crf_decode(potentials=out, transition_params=trans, sequence_length=lengths)
    return (loss, out, trans, pred_ids)

