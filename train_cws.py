#!coding=utf8
from __future__ import print_function
import sys
import os
import codecs
from itertools import izip
import tensorflow as tf
from argparse import ArgumentParser

from tagger import train


class FlushFile:
    """
    A wrapper for File, allowing users see result immediately.
    """
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


def read_segmented_text(path, bigram=False, word_window = 4):
    """
    Read training data.
    """
    if bigram:
        data = ([], [], [], [], [], [])
    elif word_window == 4:
        data = ([], [], [], [], [], [], [], [], [], [], [], [])
    elif word_window == 3:
        data = ([], [], [], [], [], [], [], [])
    elif word_window == 2:
        data = ([], [], [], [], [])
    else:
        data = ([], [])

    for l in codecs.open(path, 'r', 'utf8'):
        l = l.strip()
        if not l:
            continue
        words = l.split()
        chars = []
        tags = []
        for w in words:
            chars.extend(list(w))
            if len(w) == 1:
                tags.append('S')
            else:
                tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
        data[0].append(chars)
        if bigram:
            chars = ['', ''] + chars + ['', '']
            data[1].append([a + b if a and b else '' for a, b in zip(chars[:-4], chars[1:])])
            data[2].append([a + b if a and b else '' for a, b in zip(chars[1:-3], chars[2:])])
            data[3].append([a + b if a and b else '' for a, b in zip(chars[2:-2], chars[3:])])
            data[4].append([a + b if a and b else '' for a, b in zip(chars[3:-1], chars[4:])])
        elif word_window > 0:
            chars = ['', '', ''] + chars + ['', '', '']
            # single char
            if word_window >= 1:
                data[1].append(chars[3:-3])
            if word_window >= 2:
                # bi chars
                data[2].append([a + b if a and b else '' for a, b in zip(chars[2:], chars[3:-3])])
                data[3].append([a + b if a and b else '' for a, b in zip(chars[3:-3], chars[4:])])
            if word_window >= 3:
                # tri chars
                data[4].append(
                    [a + b + c if a and b and c else '' for a, b, c in zip(chars[1:], chars[2:], chars[3:-3])])
                data[5].append(
                    [a + b + c if a and b and c else '' for a, b, c in zip(chars[2:], chars[3:-3], chars[4:])])
                data[6].append(
                    [a + b + c if a and b and c else '' for a, b, c in zip(chars[3:-3], chars[4:], chars[5:])])
            if word_window >= 4:
                # four chars
                data[7].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                                zip(chars[0:], chars[1:], chars[2:], chars[3:-3])])
                data[8].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                                zip(chars[1:], chars[2:], chars[3:-3], chars[4:])])
                data[9].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                                zip(chars[2:], chars[3:-3], chars[4:], chars[5:])])
                data[10].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                                 zip(chars[3:-3], chars[4:], chars[5:], chars[6:])])
        data[-1].append(tags)
    return data


def create_output(seqs, stags):
    """
    Create final output from characters and BMES tags.
    """
    output = []
    for seq, stag in izip(seqs, stags):
        new_sen = []
        for c, tag in izip(seq, stag):
            new_sen.append(c)
            if tag == 'S' or tag == 'E':
                new_sen.append(' ')
        output.append(''.join(new_sen))
    return output


def evaluator(data, output_dir, output_flag):
    """
    Evaluate presion, recall and F1.
    """
    seqs, gold_stags, pred_stags = data
    assert len(seqs) == len(gold_stags) == len(pred_stags)
    # Create and open temp files.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ref_path = os.path.join(output_dir, '%s.ref' % output_flag)
    pred_path = os.path.join(output_dir, '%s.pred' % output_flag)
    score_path = os.path.join(output_dir, '%s.score' % output_flag)
    # Empty words file
    temp_path = os.path.join(output_dir, '%s.temp' % output_flag)

    ref_file = codecs.open(ref_path, 'w', 'utf8')
    pred_file = codecs.open(pred_path, 'w', 'utf8')
    for l in create_output(seqs, gold_stags):
        print(l, file=ref_file)
    for i, l in enumerate(create_output(seqs, pred_stags)):
        print(l, file=pred_file)
    ref_file.close()
    pred_file.close()

    os.system('echo > %s' % temp_path)
    os.system('%s  %s %s %s > %s' % ('./score.perl', temp_path, ref_path, pred_path, score_path))
    # Sighan evaluation results
    os.system('tail -n 7 %s > %s' % (score_path, temp_path))
    eval_lines = [l.rstrip() for l in codecs.open(temp_path, 'r', 'utf8')]
    # Remove temp files.
    os.remove(ref_path)
    os.remove(pred_path)
    os.remove(score_path)
    os.remove(temp_path)
    # Precision, Recall and F1 score
    return (float(eval_lines[1].split(':')[1]),
            float(eval_lines[0].split(':')[1]),
            float(eval_lines[2].split(':')[1]))


if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)

    parser = ArgumentParser()
    parser.add_argument('--training_path', dest='training_path', default='data/datasets/sighan2005-pku/train.txt')
    parser.add_argument('--dev_path', dest='dev_path', default='data/datasets/sighan2005-pku/dev.txt')
    parser.add_argument('--test_path', dest='test_path', default='data/datasets/sighan2005-pku/test.txt')
    parser.add_argument('--pre_trained_emb_path', dest='pre_trained_emb_path', default=None)
    parser.add_argument('--pre_trained_word_emb_path', dest='pre_trained_word_emb_path', default=None)
    parser.add_argument('--model_root', dest='model_root', default='model-pku')
    parser.add_argument('--emb_size', dest='emb_size', type=int, default=200)
    parser.add_argument('--word_window', dest='word_window', type=int, default=0)
    parser.add_argument('--hidden_layers', dest='hidden_layers', type=int, default=5)
    parser.add_argument('--channels', dest='channels', type=int, default=200)
    parser.add_argument('--kernel_size', dest='kernel_size', type=int, default=3)
    parser.add_argument('--word_emb_size', dest='word_emb_size', type=int, default=50)
    parser.add_argument('--use_bn', dest='use_bn', type=int, default=0)
    parser.add_argument('--use_wn', dest='use_wn', type=int, default=1)
    parser.add_argument('--dropout_emb', dest='dropout_emb', type=float, default=0.2)
    parser.add_argument('--dropout_hidden', dest='dropout_hidden', type=float, default=0.2)
    parser.add_argument('--active_type', dest='active_type', default='glu')
    parser.add_argument('--lamd', dest='lamd', type=float, default=0)
    parser.add_argument('--fix_word_emb', dest='fix_word_emb', type=int, default=0)
    parser.add_argument('--use_crf', dest='use_crf', type=int, default=1)
    parser.add_argument('--optimizer', dest='optimizer', default='adam_0.001')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=1000)
    parser.add_argument('--max_epoches', dest='max_epoches', type=int, default=100)

    args = parser.parse_args()
    print(args)

    train_data, dev_data, test_data = (read_segmented_text(args.training_path, word_window=args.word_window),
                                       read_segmented_text(args.dev_path, word_window=args.word_window),
                                       read_segmented_text(args.test_path, word_window=args.word_window))

    sess = tf.Session()
    train(sess=sess,
          train_data=train_data,
          dev_data=dev_data,
          test_data=test_data,
          model_dir=args.model_root + '/models',
          log_dir=args.model_root + '/logs',
          emb_size=args.emb_size,
          word_emb_size=args.word_emb_size,
          hidden_layers=args.hidden_layers,
          channels=[args.channels] * args.hidden_layers,
          kernel_size=args.kernel_size,
          use_bn=args.use_bn,
          use_wn=args.use_wn,
          active_type=args.active_type,
          batch_size=args.batch_size,
          use_crf=args.use_crf,
          lamd=args.lamd,
          dropout_emb=args.dropout_emb,
          dropout_hidden=args.dropout_hidden,
          optimizer=args.optimizer,
          evaluator=evaluator,
          eval_batch_size=args.eval_batch_size,
          print_freq=50,
          pre_trained_emb_path=args.pre_trained_emb_path,
          pre_trained_word_emb_path=args.pre_trained_word_emb_path,
          fix_word_emb=args.fix_word_emb,
          max_epoches=args.max_epoches,
          scope='CWS')
    sess.close()
