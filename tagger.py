from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.crf as crf
import time
import codecs
import os
import cPickle as pickle
import numpy as np
from itertools import izip

INT_TYPE = np.int32
FLOAT_TYPE = np.float32

# If the max length of a batch exceed MAX_LENGTH, 
# we shrink the batch size to prevent out of memory.
MAX_LENGTH = 200


################################################################################
#                              GRAPH BUILDING                                  #
################################################################################
def build_input_graph(vocab_size, emb_size, word_vocab_size, word_emb_size, word_window_size, scope=None):
    """
    Gather embeddings from lookup tables.
    """
    with tf.variable_scope(scope, initializer=tf.uniform_unit_scaling_initializer()):
        seq_ids = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='seq_ids')
        seq_word_ids = [tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='seq_feature_%d_ids' % i)
                         for i in range(word_window_size)]
        embeddings = tf.get_variable('embeddings', [vocab_size, emb_size])
        embedding_output = tf.nn.embedding_lookup([embeddings], seq_ids)
        word_outputs = []
        word_embeddings = tf.get_variable('word_embeddings', [word_vocab_size, word_emb_size])
        for i in range(word_window_size):
            word_outputs.append(tf.nn.embedding_lookup([word_embeddings], seq_word_ids[i]))

        return seq_ids, seq_word_ids, tf.concat([embedding_output] + word_outputs, 2, 'inputs')


def build_tagging_graph(inputs, hidden_layers, channels, num_tags, use_crf, lamd, dropout_emb, dropout_hidden, kernel_size,
                        use_bn, use_wn, active_type, scope=None):
    """
    Build a deep neural model for sequence tagging.
    """
    with tf.variable_scope(scope, initializer=tf.uniform_unit_scaling_initializer()):
        stag_ids = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='stag_ids')
        seq_lengths = tf.placeholder(dtype=INT_TYPE, shape=[None], name='seq_lengths')

        # Default is not train.
        is_train = tf.placeholder(dtype=tf.bool, shape=[], name='is_train')

        masks = tf.cast(tf.sequence_mask(seq_lengths), FLOAT_TYPE)
 
        # Dropout on embedding output.
        if dropout_emb:
            inputs = tf.cond(is_train,
                             lambda: tf.nn.dropout(inputs, 1 - dropout_emb),
                             lambda: inputs)

        hidden_output = inputs
        pre_channels = inputs.get_shape()[-1].value
        for i in xrange(hidden_layers):

            k = kernel_size
            cur_channels = channels[i]
            filter_w = tf.get_variable('filter_w_%d' % i, shape=[k, pre_channels, cur_channels], dtype=FLOAT_TYPE)
            filter_v = tf.get_variable('filter_v_%d' % i, shape=[k, pre_channels, cur_channels], dtype=FLOAT_TYPE)
            bias_b = tf.get_variable('bias_b_%d' % i, shape=[cur_channels],
                                     initializer=tf.zeros_initializer(dtype=FLOAT_TYPE))
            bias_c = tf.get_variable('bias_c_%d' % i, shape=[cur_channels],
                                     initializer=tf.zeros_initializer(dtype=FLOAT_TYPE))

            # Weight normalization.
            if use_wn:
                epsilon = 1e-12
                g_w = tf.get_variable('g_w_%d' % i, shape=[k, 1, cur_channels], dtype=FLOAT_TYPE)
                g_v = tf.get_variable('g_v_%d' % i, shape=[k, 1, cur_channels], dtype=FLOAT_TYPE)
                # Perform wn
                filter_w = g_w * filter_w / (tf.sqrt(tf.reduce_sum(filter_w ** 2, 1, keep_dims=True)) + epsilon)
                filter_v = g_v * filter_v / (tf.sqrt(tf.reduce_sum(filter_v ** 2, 1, keep_dims=True)) + epsilon)

            w = tf.nn.conv1d(hidden_output, filter_w, 1, 'SAME') + bias_b
            v = tf.nn.conv1d(hidden_output, filter_v, 1, 'SAME') + bias_c

            if use_bn:
                w = layers.batch_norm(inputs=v, decay=0.9, is_training=is_train, center=True, scale=True,
                                      scope='BatchNorm_w_%d' % i)
                v = layers.batch_norm(inputs=w, decay=0.9, is_training=is_train, center=True, scale=True,
                                      scope='BatchNorm_v_%d' % i)

            if active_type == 'glu':
                hidden_output = w * tf.nn.sigmoid(v)
            elif active_type == 'relu':
                hidden_output = tf.nn.relu(w)
            elif active_type == 'gtu':
                hidden_output = tf.tanh(w) * tf.nn.sigmoid(v)
            elif active_type == 'tanh':
                hidden_output = tf.tanh(w)
            elif active_type == 'linear':
                hidden_output = w
            elif active_type == 'bilinear':
                hidden_output = w * v

            # Dropout on hidden output.
            if dropout_hidden:
                hidden_output = tf.cond(is_train,
                                        lambda: tf.nn.dropout(hidden_output, 1 - dropout_hidden),
                                        lambda: hidden_output
                                        )

            pre_channels = cur_channels

        # Un-scaled log probabilities.
        scores = layers.fully_connected(hidden_output, num_tags, tf.identity)

        if use_crf:
            cost, transitions = crf.crf_log_likelihood(inputs=scores, tag_indices=stag_ids,
                                                       sequence_lengths=seq_lengths)
            cost = - tf.reduce_mean(cost)
        else:
            reshaped_scores = tf.reshape(scores, [-1, num_tags])
            reshaped_stag_ids = tf.reshape(stag_ids, [-1])
            real_distribution = layers.one_hot_encoding(reshaped_stag_ids, num_tags)
            cost = tf.nn.softmax_cross_entropy_with_logits(reshaped_scores, real_distribution)
            cost = tf.reduce_sum(tf.reshape(cost, tf.shape(stag_ids)) * masks) / tf.cast(tf.shape(inputs)[0],
                                                                                         FLOAT_TYPE)

        # Calculate L2 penalty.
        l2_penalty = 0
        if lamd > 0:
            for v in tf.trainable_variables():
                if '/B:' not in v.name and '/biases:' not in v.name:
                    l2_penalty += lamd * tf.nn.l2_loss(v)
        train_cost = cost + l2_penalty

        # Summary cost.
        tf.summary.scalar('cost', cost)

        summaries = tf.summary.merge_all()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            with tf.control_dependencies([updates]):
                cost = tf.identity(cost)

        return stag_ids, seq_lengths, is_train, cost, train_cost, scores, summaries


def inference(scores, sequence_lengths=None, transitions=None):
    """
    Inference label sequence given scores.
    If transitions is given, then perform veterbi search, else perform greedy search.

    Args:
        scores: A numpy array with shape (batch, max_length, num_tags).
        sequence_lengths: A numpy array with shape (batch,).
        transitions: A numpy array with shape (num_tags, num_tags).

    Returns:
        A numpy array with shape (batch, max_length).

    """

    if transitions is None:
        return np.argmax(scores, 2)
    else:
        assert sequence_lengths is not None and scores.shape[2] == transitions.shape[0] == transitions.shape[1]
        paths = np.zeros(scores.shape[:2], dtype=INT_TYPE)
        for i in xrange(scores.shape[0]):
            tag_score, length = scores[i], sequence_lengths[i]
            if length == 0:
                continue
            path, _ = crf.viterbi_decode(tag_score[:length], transitions)
            paths[i, :length] = path
        return paths


def train(sess, train_data, dev_data, test_data, model_dir, log_dir, emb_size, word_emb_size, optimizer,
          hidden_layers, channels, kernel_size, active_type, use_bn, use_wn, use_crf, lamd, dropout_emb, 
          dropout_hidden, evaluator, batch_size, eval_batch_size, pre_trained_emb_path=None, 
          pre_trained_word_emb_path=None, max_epoches=100, print_freq=50, scope='tagging'):

    assert len(channels) == hidden_layers

    # Parse optimization method and parameters.
    optimizer = optimizer.split('_')
    optimizer_name = optimizer[0]
    optimizer_options = [eval(i) for i in optimizer[1:]]
    optimizer = {
        'sgd': tf.train.GradientDescentOptimizer,
        'adadelta': tf.train.AdadeltaOptimizer,
        'adam': tf.train.AdamOptimizer,
        'mom': tf.train.MomentumOptimizer
    }[optimizer_name](*optimizer_options)

    print('Preparing data...', end='')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    mappings_path = os.path.join(model_dir, 'mappings.pkl')
    parameters_path = os.path.join(model_dir, 'parameters.pkl')

    pre_trained = {}
    if pre_trained_emb_path and os.path.isfile(pre_trained_emb_path):
        for l in codecs.open(pre_trained_emb_path, 'r', 'utf8'):
            we = l.split()
            if len(we) == emb_size + 1:
                w, e = we[0], np.array(map(float, we[1:]))
                pre_trained[w] = e

    # Load word embeddings
    pre_trained_word = {}
    if pre_trained_word_emb_path and os.path.isfile(pre_trained_word_emb_path):
        for l in codecs.open(pre_trained_word_emb_path, 'r', 'utf8', 'ignore'):
            we = l.split()
            if len(we) == word_emb_size + 1:
                w, e = we[0], np.array(map(float, we[1:]))
                pre_trained_word[w] = e

    # Load or create mappings.
    if os.path.isfile(mappings_path):
        item2id, id2item, tag2id, id2tag, word2id, id2word = pickle.load(open(mappings_path, 'r'))
    else:
        item2id, id2item = create_mapping(create_dic(train_data[0], add_unk=True, add_pad=True))
        tag2id, id2tag = create_mapping(create_dic(train_data[-1]))

        words = []
        for t in train_data[1:-1]:
            words.extend(t)
        for t in dev_data[1:-1]:
            words.extend(t)
        for t in test_data[1:-1]:
            words.extend(t)
        word_dic = create_dic(words, add_unk=True, add_pad=True)
        for k in word_dic.keys():
            if k not in pre_trained_word and k != '<UNK>' and k != '<PAD>':
                word_dic.pop(k)
        word2id, id2word = create_mapping(word_dic)
        # Save the mappings to disk.
        pickle.dump((item2id, id2item, tag2id, id2tag, word2id, id2word), open(mappings_path, 'w'))

    # Hyper parameters.
    word_window_size = len(train_data) - 2
    parameters = {
        'vocab_size': len(item2id),
        'emb_size': emb_size,
        'word_window_size': word_window_size,
        'word_vocab_size': len(word2id),
        'word_emb_size': word_emb_size,
        'hidden_layers': hidden_layers,
        'channels': channels,
        'kernel_size': kernel_size,
        'use_bn': use_bn,
        'use_wn': use_wn,
        'num_tags': len(tag2id),
        'use_crf': use_crf,
        'lamd': lamd,
        'dropout_emb': dropout_emb,
        'dropout_hidden': dropout_hidden,
        'active_type': active_type
    }

    if os.path.isfile(parameters_path):
        parameters_old = pickle.load(open(parameters_path, 'r'))
        if parameters != parameters_old:
            raise Exception('Network parameters are not consistent!')
    else:
        pickle.dump(parameters, open(parameters_path, 'w'))

    # Convert data to corresponding ids.
    train_data_ids = data_to_ids(
        train_data, [item2id] + [word2id] * word_window_size + [tag2id]
    )
    dev_data_ids = data_to_ids(
        dev_data, [item2id] + [word2id] * word_window_size + [tag2id]
    )
    test_data_ids = data_to_ids(
        test_data, [item2id] + [word2id] * word_window_size + [tag2id]
    )
    print('Finished.')

    print("Start building the network...", end='')
    seq_ids_pl, seq_other_ids_pls, inputs = build_input_graph(vocab_size=parameters['vocab_size'],
                                                              emb_size=parameters['emb_size'],
                                                              word_window_size=parameters['word_window_size'],
                                                              word_vocab_size=parameters['word_vocab_size'],
                                                              word_emb_size=parameters['word_emb_size'],
                                                              scope=scope)
    stag_ids_pl, seq_lengths_pl, is_train_pl, cost_op, train_cost_op, scores_op, summary_op = \
        build_tagging_graph(inputs=inputs,
                            num_tags=parameters['num_tags'],
                            use_crf=parameters['use_crf'],
                            lamd=parameters['lamd'],
                            dropout_emb=parameters['dropout_emb'],
                            dropout_hidden=parameters['dropout_hidden'],
                            hidden_layers=parameters['hidden_layers'],
                            channels=parameters['channels'],
                            kernel_size=parameters['kernel_size'],
                            use_bn=parameters['use_bn'],
                            use_wn=parameters['use_wn'],
                            active_type=parameters['active_type'],
                            scope=scope)

    # An inner tag function for evalution on dev and test data.
    def tag(data):
        iter = data_iterator(data, batch_size=eval_batch_size, shuffle=False)
        output = []
        for batch in iter:
            batch = create_input(batch)
            seq_ids, seq_other_ids_list, seq_lengths = batch[0], batch[1: -1], batch[-1]
            feed_dict = {seq_ids_pl: seq_ids.astype(INT_TYPE),
                         seq_lengths_pl: seq_lengths.astype(INT_TYPE),
                         is_train_pl: False}
            assert len(seq_other_ids_pls) == len(seq_other_ids_list)
            for pl, v in zip(seq_other_ids_pls, seq_other_ids_list):
                feed_dict[pl] = v.astype(INT_TYPE)
            scores = sess.run(scores_op, feed_dict)
            if parameters['use_crf']:
                with tf.variable_scope(scope, reuse=True):
                    stag_ids = inference(scores, seq_lengths, tf.get_variable('transitions').eval(session=sess))
            else:
                stag_ids = inference(scores, seq_lengths)
            for stag_id, length in izip(stag_ids, seq_lengths):
                output.append([id2tag[t] for t in stag_id[:length]])
        return output

    def summary(name, dtype=FLOAT_TYPE):
        value = tf.placeholder(dtype, shape=[])
        return value, tf.summary.scalar(name, value)

    dev_f1_pl, dev_summary_op = summary('dev f1')
    test_f1_pl, test_summary_op = summary('test f1')

    # Clip gradients and apply.
    grads_and_vars = optimizer.compute_gradients(loss=train_cost_op, var_list=tf.trainable_variables())
    grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]
    grads_summary_op = tf.summary.histogram('grads', tf.concat([tf.reshape(g, [-1]) for g, _ in grads_and_vars], 0))
    grads_norm = tf.sqrt(sum([tf.reduce_sum(tf.pow(g, 2)) for g, _ in grads_and_vars]))
    grads_and_vars = [(g / (tf.reduce_max([grads_norm, 5]) / 5), v) for g, v in grads_and_vars]

    train_op = optimizer.apply_gradients(grads_and_vars)

    # Variables for recording training procedure
    best_epoch = tf.get_variable('best_epoch',
                                 shape=[],
                                 initializer=tf.zeros_initializer(),
                                 trainable=False,
                                 dtype=INT_TYPE)
    best_dev_score = tf.get_variable('best_dev_score',
                                     shape=[],
                                     initializer=tf.zeros_initializer(),
                                     trainable=False,
                                     dtype=FLOAT_TYPE)
    best_test_score = tf.get_variable('best_test_score',
                                      shape=[],
                                      initializer=tf.zeros_initializer(),
                                      trainable=False,
                                      dtype=FLOAT_TYPE)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(log_dir + '/summaries', sess.graph)

    print('Finished.')
    print('Start training the network...')
    sess.run(init_op)

    start_time_begin = time.time()
    global_step = 0

    try:
        checkpoint = tf.train.latest_checkpoint(model_dir)
        saver.restore(sess, checkpoint)
        print('Restore model from %s.' % checkpoint)
    except (tf.errors.DataLossError, TypeError, Exception):
        # Failed to restore model from disk. Load pre-trained embeddings.
        # Load character embeddings.
        with tf.variable_scope(scope, reuse=True):
            embeddings = tf.get_variable('embeddings')
        value = sess.run(embeddings)
        count = 0
        for item in item2id:
            item_id = item2id[item]
            if item in pre_trained:
                value[item_id] = pre_trained[item]
                count += 1
        # Run assign op.
        sess.run(embeddings.assign(value))
        del(pre_trained)
        print('%d of %d character embeddings were loaded from pre-trained.' % (count, len(item2id)))

        # Load word embeddings.
        with tf.variable_scope(scope, reuse=True):
            word_embeddings = tf.get_variable('word_embeddings')
        value = sess.run(word_embeddings)
        count = 0
        for item in word2id:
            item_id = word2id[item]
            if item in pre_trained_word:
                value[item_id] = pre_trained_word[item]
                count += 1
        # Run assign op.
        sess.run(word_embeddings.assign(value))
        del(pre_trained_word)
        print('%d of %d word embeddings were loaded from pre-trained.' % (count, len(word2id)))

    start_epoch, best_dev_f1 = sess.run((best_epoch, best_dev_score))

    for epoch in range(start_epoch + 1, max_epoches + 1):
        print('Starting epoch %d...' % epoch)
        start_time = time.time()
        loss_ep = 0
        n_step = 0
        iterator = data_iterator(train_data_ids, batch_size, shuffle=True)
        for batch in iterator:
            batch = create_input(batch)
            seq_ids, seq_other_ids_list, stag_ids, seq_lengths = batch[0], batch[1: -2], batch[-2], batch[-1]
            feed_dict = {seq_ids_pl: seq_ids.astype(INT_TYPE),
                         stag_ids_pl: stag_ids.astype(INT_TYPE),
                         seq_lengths_pl: seq_lengths.astype(INT_TYPE),
                         is_train_pl: True}
            assert len(seq_other_ids_pls) == len(seq_other_ids_list)
            for pl, v in zip(seq_other_ids_pls, seq_other_ids_list):
                feed_dict[pl] = v
            # feed_dict.update(drop_feed_dict)  # enable noise input
            loss, summaries, grads_summaries, _ = sess.run([cost_op, summary_op, grads_summary_op, train_op],
                                                           feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
            global_step += 1
            summary_writer.add_summary(summaries, global_step)
            summary_writer.add_summary(grads_summaries, global_step)

            # Show training information
            if global_step % print_freq == 0:
                print('  Step %d, current cost %.6f, average cost %.6f' % (global_step, loss, loss_ep / n_step))

        loss_ep = loss_ep / n_step
        print('Epoch %d finished. Time: %ds Cost: %.6f' % (epoch, time.time() - start_time, loss_ep))

        # Evaluate precision, recall and f1 with an external script.
        dev_pre, dev_rec, dev_f1 = evaluator((dev_data[0], dev_data[-1], tag(dev_data_ids[:-1])),
                                             log_dir + '/dev',
                                             epoch)
        test_pre, test_rec, test_f1 = evaluator((test_data[0], test_data[-1], tag(test_data_ids[:-1])),
                                                log_dir + '/test',
                                                epoch)
        summary_writer.add_summary(sess.run(dev_summary_op, {dev_f1_pl: dev_f1}), epoch)
        summary_writer.add_summary(sess.run(test_summary_op, {test_f1_pl: test_f1}), epoch)
        print("Dev   precision / recall / f1 score: %.2f / %.2f / %.2f" %
              (dev_pre * 100, dev_rec * 100, dev_f1 * 100))
        print("Test  precision / recall / f1 score: %.2f / %.2f / %.2f" %
              (test_pre * 100, test_rec * 100, test_f1 * 100))
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1

            sess.run((tf.assign(best_epoch, epoch),
                      tf.assign(best_dev_score, dev_f1),
                      tf.assign(best_test_score, test_f1)))

            path = saver.save(sess, model_dir + '/model', epoch)
            print('New best score on dev.')
            print('Save model at %s.' % path)

    print('Finished.')
    print('Total training time: %fs.' % (time.time() - start_time_begin))


def tag(sess, data_iter, model_dir, scope='tagging'):
    """A tagging function.

    Args:
        sess: A tensorflow session.
        data_iter: A iterator for generate batches.
        model_dir: Where is the model.
        scope: Tensorflow namescope.

    Returns:
        A generator for tagging result.

    """
    mappings_path = os.path.join(model_dir, 'mappings.pkl')
    parameters_path = os.path.join(model_dir, 'parameters.pkl')
    item2id, id2item, tag2id, id2tag, word2id, id2word = pickle.load(open(mappings_path, 'r'))
    parameters = pickle.load(open(parameters_path))
    print(parameters)
    print('Building input graph...', end='')
    seq_ids_pl, seq_other_ids_pls, inputs = build_input_graph(vocab_size=parameters['vocab_size'],
                                                              emb_size=parameters['emb_size'],
                                                              word_window_size=parameters['word_window_size'],
                                                              word_vocab_size=parameters['word_vocab_size'],
                                                              word_emb_size=parameters['word_emb_size'],
                                                              scope=scope)
    print('Finished.')
    print('Building tagging graph...', end='')
    stag_ids_pl, seq_lengths_pl, is_train_pl, cost_op, train_cost_op, scores_op, summary_op = \
        build_tagging_graph(inputs=inputs,
                            num_tags=parameters['num_tags'],
                            use_crf=parameters['use_crf'],
                            lamd=parameters['lamd'],
                            dropout_emb=parameters['dropout_emb'],
                            dropout_hidden=parameters['dropout_hidden'],
                            hidden_layers=parameters['hidden_layers'],
                            channels=parameters['channels'],
                            kernel_size=parameters['kernel_size'],
                            use_bn=parameters['use_bn'],
                            use_wn=parameters['use_wn'],
                            active_type=parameters['active_type'],
                            scope=scope)
    print('Finished.')
    print('Initializing variables...', end='')
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print('Finished.')
    print('Reloading parameters...', end='')
    saver = tf.train.Saver(tf.global_variables())
    checkpoint = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, checkpoint)
    print('Finished.')

    tf.get_variable_scope().reuse_variables()
    output = []
    for data in data_iter:
        batch = data_to_ids(data, [item2id] + [word2id] * parameters['word_window_size'])
        batch = create_input(batch)
        seq_ids, seq_other_ids_list, seq_lengths = batch[0], batch[1: -1], batch[-1]
        feed_dict = {seq_ids_pl: seq_ids.astype(INT_TYPE),
                     seq_lengths_pl: seq_lengths.astype(INT_TYPE),
                     is_train_pl: False}
        assert len(seq_other_ids_pls) == len(seq_other_ids_list)
        for pl, v in zip(seq_other_ids_pls, seq_other_ids_list):
            feed_dict[pl] = v.astype(INT_TYPE)
        scores = sess.run(scores_op, feed_dict)
        if parameters['use_crf']:
            with tf.variable_scope(scope, reuse=True):
                stag_ids = inference(scores, seq_lengths, tf.get_variable('transitions').eval(session=sess))
        else:
            stag_ids = inference(scores, seq_lengths)
        for seq, stag_id, length in izip(data[0], stag_ids, seq_lengths):
            output.append((seq, [id2tag[t] for t in stag_id[:length]]))
        yield zip(*output)
        output = []


################################################################################
#                                 DATA UTILS                                   #
################################################################################
def create_dic(item_list, add_unk=False, add_pad=False):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dic = {}
    for items in item_list:
        for item in items:
            if item not in dic:
                dic[item] = 1
            else:
                dic[item] += 1
    # Make sure that <PAD> have a id 0.
    if add_pad:
        dic['<PAD>'] = 1e20
    # If specified, add a special item <UNK>.
    if add_unk:
        dic['<UNK>'] = 1e10
    return dic


def create_mapping(items):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    if type(items) is dict:
        sorted_items = sorted(items.items(), key=lambda x: (-x[1], x[0]))
        id2item = {i: v[0] for i, v in enumerate(sorted_items)}
        item2id = {v: k for k, v in id2item.items()}
        return item2id, id2item
    elif type(items) is list:
        id2item = {i: v for i, v in enumerate(items)}
        item2id = {v: k for k, v in id2item.items()}
        return item2id, id2item


def create_input(batch):
    """
    Take each sentence data in batch and return an input for
    the training or the evaluation function.
    """
    assert len(batch) > 0
    lengths = [len(seq) for seq in batch[0]]
    max_len = max(2, max(lengths))
    ret = []
    for d in batch:
        dd = []
        for seq_id, pos in izip(d, lengths):
            assert len(seq_id) == pos
            pad = [0] * (max_len - pos)
            dd.append(seq_id + pad)
        ret.append(np.array(dd))
    ret.append(np.array(lengths))
    return ret


def data_to_ids(data, mappings):
    """
    Map text data to ids.
    """

    def map_seq(seqs, mapping):
        return [[mapping[item] if item in mapping else mapping['<UNK>']
                 for item in seq] for seq in seqs]

    ret = []
    for d, m in izip(data, mappings):
        ret.append(map_seq(d, m))
    return tuple(ret)


def data_iterator(inputs, batch_size, shuffle=True):
    """
    A simple iterator for generate mini batches.
    """
    assert len(inputs) > 0
    assert all([len(item) == len(inputs[0]) for item in inputs])
    inputs = zip(*inputs)
    if shuffle:
        np.random.shuffle(inputs)

    batch = []
    bs = batch_size
    for d in inputs:
        if len(d[0]) > MAX_LENGTH:
            bs = max(1, min(batch_size * MAX_LENGTH / len(d[0]), bs))
        if len(batch) < bs:
            batch.append(d)
        else:
            yield zip(*batch)
            batch = [d]
            if len(d[0]) < MAX_LENGTH:
                bs = batch_size
            else:
                bs = max(1, batch_size * MAX_LENGTH / len(d[0]))
    if batch:
        yield zip(*batch)
