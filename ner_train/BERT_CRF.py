import os
import pickle
import codecs
import collections
import tensorflow as tf
from bert import tokenization
from ner_train import train_helper
from tensorflow.contrib import crf
from bert import modeling, optimization
from tensorflow.contrib.layers.python.layers import initializers


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, separator=' '):
        """Reads a BIO data."""
        wts = []
        words = []
        tags = []
        with codecs.open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                contends = line.strip()
                if contends:
                    tokens = contends.split(separator)
                    words.append(tokens[0])
                    tags.append(tokens[1])
                else:
                    wts.append([' '.join(words), ' '.join(tags)])
                    words = []
                    tags = []
        return wts


class NerProcessor(DataProcessor):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def get_train_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "naso_train.txt")), "train"
        )

    def get_dev_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "naso_dev.txt")), "dev"
        )

    def get_test_examples(self, data_dir):
        return self._create_example(
            self._read_data(os.path.join(data_dir, "naso_test.txt")), "test")

    def get_labels(self):
        self.labels = ['O', 'B-L', 'I-L', 'S-L',
                            'B-P', 'I-P', 'S-P',
                            'B-S', 'I-S', 'S-S',
                            'B-A', 'I-A', 'S-A',
                            'B-V', 'I-V', 'S-V',
                            'S-C', "X"]

        return self.labels

    def _create_example(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])

            examples.append(InputExample(guid=guid, text_a=text, label=label))
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :return:
    """
    label_map = {}
    # 1表示从1开始对label进行index化
    for (i, label) in enumerate(label_list, 1):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    label_a = example.label.split(' ')
    if len(tokens_a) > max_seq_length:
        tokens_a = tokens_a[0:max_seq_length]
        label_a = label_a[0:max_seq_length]
    tokens = []
    label_ids = []
    segment_ids = []
    for index, token in enumerate(tokens_a):
        tokens.append(token)
        label_ids.append(label_map[label_a[index]])
        segment_ids.append(0)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        label_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature


def file_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())


def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
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
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,segment_ids, labels,
                 num_labels, use_one_hot_embeddings,dropout_rate=0.9):
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

    if is_training:
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
        predict = tf.tanh(tf.nn.xw_plus_b(output, W, b))
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


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate, num_train_steps, num_warmup_steps, args):
    """
    构建模型
    :param bert_config:
    :param num_labels:
    :param init_checkpoint:
    :param learning_rate:
    :param num_train_steps:
    :param num_warmup_steps:
    :return:
    """

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        print('shape of input_ids', input_ids.shape)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # 使用参数构建模型,input_idx 就是输入的样本idx表示，label_ids 就是标签的idx表示
        total_loss, logits, trans, pred_ids = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, False, args.dropout_rate)

        tvars = tf.trainable_variables()
        # 加载BERT模型
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                 modeling.get_assignment_map_from_checkpoint(tvars,
                                                             init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(
                 total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
            hook_dict = {}
            hook_dict['loss'] = total_loss
            hook_dict['global_steps'] = tf.train.get_or_create_global_step()
            logging_hook = tf.train.LoggingTensorHook(
                hook_dict, every_n_iter=args.save_summary_steps)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                training_hooks=[logging_hook])

        elif mode == tf.estimator.ModeKeys.EVAL:
            # 针对NER ,进行了修改
            def metric_fn(label_ids, pred_ids):
                return {
                    "eval_loss": tf.metrics.mean_squared_error(labels=label_ids, predictions=pred_ids),
                }

            eval_metrics = metric_fn(label_ids, pred_ids)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=pred_ids
            )
        return output_spec

    return model_fn


def train(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_map

    processors = {
        "ner": NerProcessor
    }
    bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_seq_length, bert_config.max_position_embeddings))

    # 在retrain 的时候，才删除上一轮产出的文件，在predicted 的时候不做clean
    if args.clean and args.do_train:
        if os.path.exists(args.output_dir):
            def del_file(path):
                ls = os.listdir(path)
                for i in ls:
                    c_path = os.path.join(path, i)
                    if os.path.isdir(c_path):
                        del_file(c_path)
                    else:
                        os.remove(c_path)

            try:
                del_file(args.output_dir)
            except Exception as e:
                print(e)
                print('pleace remove the files of output dir and data.conf')
                exit(-1)

    #check output dir exists
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    processor = processors[args.ner](args.output_dir)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    run_config = tf.estimator.RunConfig(
        model_dir=args.output_dir,
        save_summary_steps=500,
        save_checkpoints_steps=args.save_checkpoints_steps,
        session_config=session_config
    )

    train_examples = None
    eval_examples = None
    num_train_steps = None
    num_warmup_steps = None

    if args.do_train and args.do_eval:
        # 加载训练数据
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(
            len(train_examples) *1.0 / args.batch_size * args.num_train_epochs)
        if num_train_steps < 1:
            raise AttributeError('training data is so small...')
        num_warmup_steps = int(num_train_steps * args.warmup_proportion)

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d" % len(train_examples))
        tf.logging.info("  Batch size = %d" % args.batch_size)
        tf.logging.info("  Num steps = %d" % num_train_steps)

        eval_examples = processor.get_dev_examples(args.data_dir)

        # 打印验证集数据信息
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d" % len(eval_examples))
        tf.logging.info("  Batch size = %d" % args.batch_size)

    label_list = processor.get_labels()
    label2id = {}
    id2label = {}

    for (i, label) in enumerate(label_list, 1):
        label2id[label] = i
        id2label[i] = label

    with codecs.open(os.path.join(args.output_dir, 'id2label.pkl'), 'wb') as i2l:
        pickle.dump(id2label, i2l)

    print(label2id)

    with codecs.open(os.path.join(args.output_dir, 'label2id.pkl'), 'wb') as l2i:
        pickle.dump(label2id, l2i)

    # 返回的model_dn 是一个函数，其定义了模型，训练，评测方法，并且使用钩子参数，加载了BERT模型的参数进行了自己模型的参数初始化过程
    # tf 新的架构方法，通过定义model_fn 函数，定义模型，然后通过EstimatorAPI进行模型的其他工作，Es就可以控制模型的训练，预测，评估工作等。
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=args.init_checkpoint,
        learning_rate=args.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        args=args)

    params = {
        'batch_size': args.batch_size
    }

    estimator = tf.estimator.Estimator(
        model_fn,
        params=params,
        config=run_config)

    if args.do_train and args.do_eval:
        # 1. 将数据转化为tf_record 数据
        train_file = os.path.join(args.output_dir, "train.tf_record")
        if not os.path.exists(train_file):
            file_based_convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer, train_file)

        # 2.读取record 数据，组成batch
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=args.max_seq_length,
            is_training=True,
            drop_remainder=True)
        # estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        eval_file = os.path.join(args.output_dir, "eval.tf_record")
        if not os.path.exists(eval_file):
            file_based_convert_examples_to_features(eval_examples, label_list,
                                                    args.max_seq_length, tokenizer, eval_file)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=False)

        # train and eval together
        # early stop hook
        early_stopping_hook = tf.contrib.estimator.stop_if_no_decrease_hook(
            estimator=estimator,
            metric_name='loss',
            max_steps_without_decrease=num_train_steps,
            eval_dir=None,
            min_steps=0,
            run_every_secs=None,
            run_every_steps=args.save_checkpoints_steps)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps,
                                            hooks=[early_stopping_hook])
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.do_predict:
        token_path = os.path.join(args.output_dir, "token_test.txt")
        if os.path.exists(token_path):
            os.remove(token_path)

        file_id2label = os.path.join(args.output_dir, 'id2label.pkl')
        if os.path.exists(file_id2label):
            with codecs.open(file_id2label, 'rb') as rf:
                id2label = pickle.load(rf)
        else:
            tf.logging.info(file_id2label, ' is not exits, please check the direction')

        predict_examples = processor.get_test_examples(args.data_dir)
        predict_file = os.path.join(args.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                args.max_seq_length, tokenizer, predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d" % len(predict_examples))
        tf.logging.info("  Batch size = %d" % args.batch_size)

        predict_drop_remainder = False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=args.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(args.output_dir, "ner_predict.txt")

        def write_result2file(writer):
            for predict_line, prediction in zip(predict_examples, result):
                idx = 0
                line = ''
                line_token = str(predict_line.text_a).split(' ')
                label_token = str(predict_line.label).split(' ')
                len_seq = len(label_token)
                if len(line_token) != len(label_token):
                    tf.logging.info(predict_line.text_a)
                    tf.logging.info(predict_line.label)
                    tf.logging.info('the length of text and label is not equal')
                    break
                for id in prediction:
                    if idx >= len_seq:
                        break
                    if id == 0:
                        curr_labels = ''
                    else:
                        curr_labels = id2label[id]
                    try:
                        line += line_token[idx] + ' ' + label_token[idx] + ' ' + curr_labels + '\n'
                    except Exception as e:
                        tf.logging.info(e)
                        tf.logging.info(predict_line.text_a)
                        tf.logging.info(predict_line.label)
                        line = ''
                        break
                    idx += 1
                writer.write(line + '\n')

        with codecs.open(output_predict_file, 'w', encoding='utf8') as f:
            write_result2file(f)

        from ner_train import utils
        evaluator = utils.Evaluator(output_predict_file, label2id)  # 初始化指标评价器
        result_all = evaluator.get_metrics(mode='all')  # 返回总的指标
        result_precise = evaluator.get_metrics(mode='precise')  # 返回每个实体的指标
        # 将各指标写到文件中
        with codecs.open(os.path.join(args.output_dir, 'predict_score.txt'), 'w', encoding='utf-8') as fd:
            fd.write(utils.sparse_result(result_all))  # 写总的指标
            fd.write(utils.sparse_result(result_precise))  # 写每个实体的指标


if __name__ == '__main__':
    args = train_helper.get_args_parser()
    train(args)



