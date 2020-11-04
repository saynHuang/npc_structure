import os
import pickle
import numpy as np
import tensorflow as tf
from ner_train.utils import plot_loss
from tensorflow.contrib.layers.python.layers import initializers


def train_model(data_dir, model_save_dir):
    initializer = initializers.xavier_initializer()
    filter_width = 3
    num_filter = n_units
    conv_depth = 3

    train_dir = os.path.join(data_dir, 'naso_id_train.pkl')
    dev_dir = os.path.join(data_dir, 'naso_id_dev.pkl')

    # 加载训练数据
    with open(train_dir, 'rb') as data:
        x_data = pickle.load(data)
        y_data = pickle.load(data)
        efficient_seq_len = pickle.load(data)
    x_data = np.array(x_data, dtype=np.int64)
    y_data = np.array(y_data, dtype=np.int64)
    efficient_seq_len = np.array(efficient_seq_len, dtype=np.int64)
    n_samples = x_data.shape[0]

    # 加载验证数据
    with open(dev_dir, 'rb') as test_data:
        x_dev = pickle.load(test_data)
        y_dev = pickle.load(test_data)
        efficient_seq_len_test = pickle.load(test_data)
    x_dev = np.array(x_dev, dtype=np.int64)
    y_dev = np.array(y_dev, dtype=np.int64)
    efficient_seq_len_test = np.array(efficient_seq_len_test, dtype=np.int64)


    # 构建graph
    with tf.name_scope('input'):
        xs = tf.placeholder(dtype=tf.int64, shape=[None, seq_len], name='input_data')  # 输入的是字的序号
        ys = tf.placeholder(dtype=tf.int64, shape=[None, seq_len], name='label')
        sequence_lengths = tf.placeholder(tf.int64, shape=[None], name='sequence_lengths')

    word_embeddings = tf.Variable(tf.random_normal(shape=[dict_size, embedding_dim]))  # 词嵌入矩阵初始化
    input_embedded = tf.nn.embedding_lookup(word_embeddings, xs)  # [n_samples, seq_len， embedding_dim]
    input_embedded = tf.nn.dropout(input_embedded, dropout_keep)

    # CNN_layer
    model_inputs = tf.expand_dims(input_embedded, 1)
    with tf.variable_scope("cnn"):
        filter_weights = tf.get_variable(name="cnn_filter",
                                         shape=[1, filter_width, embedding_dim, num_filter],
                                         initializer=initializer)

        """
        shape of input = [batch, in_height, in_width, in_channels]
        shape of filter = [filter_height, filter_width, in_channels, out_channels]
        """
        layerInput = tf.nn.conv2d(model_inputs,
                                  filter_weights,
                                  strides=[1, 1, 1, 1],
                                  padding="SAME",
                                  name="init_layer")
        finalOutFromLayers = []
        totalWidthForLastDim = 0
        for i in range(conv_depth):
            isLast = True if i == (conv_depth - 1) else False
            with tf.variable_scope("conv_layer_%d" % (i+1), reuse=tf.AUTO_REUSE):
                w = tf.get_variable(name="conv_W",
                                    shape=[1, filter_width, num_filter, num_filter],
                                    initializer=initializer)
                b = tf.get_variable("conv_b", shape=[num_filter])
                conv = tf.nn.atrous_conv2d(layerInput,
                                           w,
                                           rate=1,
                                           padding="SAME")
                conv = tf.nn.bias_add(conv, b)
                conv = tf.nn.relu(conv)
                if isLast:
                    finalOutFromLayers.append(conv)
                    totalWidthForLastDim += num_filter
                layerInput = conv
        finalOut = tf.concat(axis=3, values=finalOutFromLayers)
        finalOut = tf.nn.dropout(finalOut, dropout_keep)

        finalOut = tf.squeeze(finalOut, [1])
        finalOut = tf.reshape(finalOut, [-1, totalWidthForLastDim])

    # Fully connected layer.
    with tf.name_scope('fully_connect'):
        W = tf.get_variable(name='fc_W', shape=[totalWidthForLastDim, n_units], dtype=tf.float32)
        b = tf.get_variable(name='fc_b', shape=[n_units], dtype=tf.float32, initializer=tf.zeros_initializer())
        predict = tf.tanh(tf.nn.xw_plus_b(finalOut, W, b))  # [n_samples*seq_len, n_units]
        out = tf.reshape(predict, shape=[-1, seq_len, n_units], name='out')

        # Linear-CRF.
        with tf.name_scope('crf'):
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(out, ys, sequence_lengths)
            cost = tf.reduce_mean(-log_likelihood, name='loss')

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(cost)

    saver = tf.train.Saver()
    x_batch, y_batch, mask_batch = get_batch(x_data, y_data, efficient_seq_len, batch_size, epochs)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        n_batch = 0
        epoch = 0
        train_loss, dev_loss = [], []
        try:
            while not coord.should_stop():
                data, label, mask = sess.run([x_batch, y_batch, mask_batch])
                sess.run(train_op, feed_dict={xs: data, ys: label, sequence_lengths: mask})
                n_batch += 1
                if n_samples//(n_batch*batch_size) == 0:
                    n_batch = 0
                    tr_loss = sess.run(cost, feed_dict={xs: x_data, ys: y_data, sequence_lengths: efficient_seq_len})
                    de_loss = sess.run(cost, feed_dict={xs: x_dev, ys: y_dev, sequence_lengths: efficient_seq_len_test})
                    train_loss.append(tr_loss)
                    dev_loss.append(de_loss)
                    epoch += 1
                    # if epoch % 10 == 0:
                    print('epoch %4d: Loss --> train:%.4f\t dev:%.4f' % (epoch, tr_loss, de_loss))
        except tf.errors.OutOfRangeError:
            sess.run(train_op, feed_dict={xs: x_dev, ys: y_dev, sequence_lengths: efficient_seq_len_test})
            print('--------------train end--------------')
        finally:
            coord.request_stop()
        coord.join(threads)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        save_path = saver.save(sess, os.path.join(model_save_dir, 'cnn_crf'))
        print("Model saved in file: %s" % save_path)

        transition_params = sess.run(transition_params, feed_dict={xs: x_data, ys: y_data, sequence_lengths: efficient_seq_len})
        with open(os.path.join(model_save_dir, 'trans_matrix.pkl'), 'wb') as trans_mat:
            pickle.dump(transition_params, trans_mat)

    plot_loss(train_loss, dev_loss, model_save_dir, 'cnn_crf')


def get_batch(data, label, mask, batch_size, epochs):
    input_squeue = tf.train.slice_input_producer([data, label, mask], num_epochs=epochs, shuffle=False)
    x_batch, y_batch, mask_batch = tf.train.batch(input_squeue, batch_size, allow_smaller_final_batch=True)
    return x_batch, y_batch, mask_batch


def load_model(data_dir, model_save_dir):
    test_dir = os.path.join(data_dir, 'naso_id_test.pkl')
    # 加载训练数据
    with open(test_dir, 'rb') as data:
        x_data = pickle.load(data)
        y_data = pickle.load(data)
        efficient_seq_len = pickle.load(data)
    x_data = np.array(x_data, dtype=np.int64)
    y_data = np.array(y_data, dtype=np.int64)
    efficient_seq_len = np.array(efficient_seq_len, dtype=np.int64)

    model_name = 'cnn_crf.meta'
    pred_name = 'cnn_crf_predict_score.txt'
    # 加载转移矩阵
    with open(os.path.join(model_save_dir, 'trans_matrix.pkl'), 'rb') as trans_mat:
        transition_matrix = pickle.load(trans_mat)

    sess = tf.Session()
    # 加载模型
    saver = tf.train.import_meta_graph(os.path.join(model_save_dir, model_name))  # 先加载meta文件，具体到文件名
    saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))  # 加载检查点文件，具体到文件夹即可
    graph = tf.get_default_graph()
    xs = graph.get_tensor_by_name('input/input_data:0')  # 获取占位符xs
    ys = graph.get_tensor_by_name('input/label:0')  # 获取占位符ys
    sequence_lengths = graph.get_tensor_by_name('input/sequence_lengths:0')
    predict = graph.get_tensor_by_name('fully_connect/out:0')

    out = sess.run(predict, feed_dict={xs: x_data, ys: y_data, sequence_lengths: efficient_seq_len})
    # viterbi解码
    viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(out, transition_matrix, efficient_seq_len)
    final_pred = sess.run(viterbi_sequence)
    sess.close()

    # return viterbi_tag
    acc, psda_acc = model_eval(final_pred, y_data)
    recall, psda_recall = model_eval(y_data, final_pred)
    f1 = 2*acc*recall/(acc+recall)
    psda_f1 = 2*psda_acc*psda_recall/(psda_acc+psda_recall)
    total_metrics = ' total  : Accuracy : %.4f    Recall : %.4f    F1 : %.4f' % (acc, recall, f1)
    p_metrics = 'entity P: Accuracy : %.4f    Recall : %.4f    F1 : %.4f' % (psda_acc[0], psda_recall[0], psda_f1[0])
    s_metrics = 'entity S: Accuracy : %.4f    Recall : %.4f    F1 : %.4f' % (psda_acc[1], psda_recall[1], psda_f1[1])
    d_metrics = 'entity A: Accuracy : %.4f    Recall : %.4f    F1 : %.4f' % (psda_acc[2], psda_recall[2], psda_f1[2])
    a_metrics = 'entity V: Accuracy : %.4f    Recall : %.4f    F1 : %.4f' % (psda_acc[3], psda_recall[3], psda_f1[3])

    print(total_metrics)
    print(p_metrics)
    print(s_metrics)
    print(d_metrics)
    print(a_metrics)

    with open(os.path.join(model_save_dir, pred_name), 'w', encoding='utf8') as f:
        f.write(total_metrics + '\n')
        f.write(p_metrics + '\n')
        f.write(s_metrics + '\n')
        f.write(d_metrics + '\n')
        f.write(a_metrics + '\n')


def model_eval(x_seq, y_seq):
    begin = [1, 4, 7, 10, 13]
    single = [3, 6, 9, 12, 15]
    [total_entity, correct_entity] = [0, 0]
    p_total, s_total, a_total, v_total = 0, 0, 0, 0
    p_entity, s_entity, d_entity, a_entity = 0, 0, 0, 0
    for index in range(len(x_seq)):
        sample_xs = x_seq[index]  # 取第index个样本
        sample_ys = y_seq[index]
        i = 0
        while i < (len(sample_xs)):
            if sample_xs[i] in begin:  # 第i个元素为begin
                start_index = i  # 实体开始索引
                end_index = 0  # 初始化结束索引为0
                if i == 49:
                    i += 1
                else:
                    for j in range(i+1, len(sample_xs)):
                        if sample_xs[j] == sample_xs[start_index] + 1:  # 后续元素值等于开始索引元素值加1
                            end_index = j  # 实体结束索引
                        else:
                            break
                    i = j

                if end_index:  # 如果结束索引不为0
                    xs_entity = sample_xs[start_index:end_index+1]  # 获取实体
                    ys_entity = sample_ys[start_index:end_index+1]
                    total_entity += 1  # 总实体数加1

                    ###########################################
                    if sample_xs[start_index] == 4:
                        p_total += 1
                    if sample_xs[start_index] == 7:
                        s_total += 1
                    if sample_xs[start_index] == 10:
                        a_total += 1
                    if sample_xs[start_index] == 13:
                        v_total += 1
                    ###########################################
                    if (xs_entity == ys_entity).all():  # 如果相等，正确实体数加1
                        correct_entity += 1

                        ###########################################
                        if sample_xs[start_index] == 4:
                            p_entity += 1
                        if sample_xs[start_index] == 7:
                            s_entity += 1
                        if sample_xs[start_index] == 10:
                            d_entity += 1
                        if sample_xs[start_index] == 13:
                            a_entity += 1
                        ###########################################

            elif sample_xs[i] in single:  # 第i个元素为begin
                xs_entity = sample_xs[i]  # 获取实体
                ys_entity = sample_ys[i]
                total_entity += 1  # 总实体数加1

                ###########################################
                if sample_xs[i] == 6:
                    p_total += 1
                if sample_xs[i] == 9:
                    s_total += 1
                if sample_xs[i] == 12:
                    a_total += 1
                if sample_xs[i] == 15:
                    v_total += 1
                ###########################################

                if xs_entity == ys_entity:  # 如果相等，正确实体数加1
                    correct_entity += 1

                    ###########################################
                    if sample_xs[i] == 6:
                        p_entity += 1
                    if sample_xs[i] == 9:
                        s_entity += 1
                    if sample_xs[i] == 12:
                        d_entity += 1
                    if sample_xs[i] == 15:
                        a_entity += 1
                    ###########################################

                i += 1

            else:
                i += 1

    p_metric = p_entity/(p_total+10e-5)
    s_metric = s_entity/(s_total+10e-5)
    d_metric = d_entity/(a_total+10e-5)
    a_metric = a_entity/(v_total+10e-5)
    return correct_entity/(total_entity+10e-5), np.array([p_metric, s_metric, d_metric, a_metric])



if __name__ == '__main__':
    embedding_dim = 50  # 词嵌入维度大小为50
    seq_len = 50  # 序列长度默认50
    dropout_keep = 0.8
    batch_size = 100
    epochs = 50
    lr = 0.001
    model_save_dir = '../ner_train/CNN_CRF_Model'

    data_dir = '../data'
    with open(os.path.join(data_dir, 'maps.pkl'), 'rb') as fr:
        dict_size = len(pickle.load(fr))  # 字典大小
        n_units = len(pickle.load(fr))  # 神经元数目，一般为标签数

    ###################################################
    train_model(data_dir, model_save_dir)
    load_model(data_dir, model_save_dir)


