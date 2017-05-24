#!coding=utf8
from __future__ import print_function
import sys, codecs
import tensorflow as tf
from argparse import ArgumentParser

from tagger import Model


class FlushFile:
    """
    A wrapper for File, allowing users see result immediately.
    """
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)

    parser = ArgumentParser()
    parser.add_argument('--task', dest='task')
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
    parser.add_argument('--reserve_all_word_emb', dest='reserve_all_word_emb', type=int, default=0)
    parser.add_argument('--use_crf', dest='use_crf', type=int, default=1)
    parser.add_argument('--optimizer', dest='optimizer', default='adam_0.001')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=100)
    parser.add_argument('--eval_batch_size', dest='eval_batch_size', type=int, default=1000)
    parser.add_argument('--max_epoches', dest='max_epoches', type=int, default=100)

    args = parser.parse_args()
    print(args)

    TASK = __import__(args.task)

    train_data, dev_data, test_data = (TASK.read_train_file(codecs.open(args.training_path, 'r', 'utf8'), word_window=args.word_window),
                                       TASK.read_train_file(codecs.open(args.dev_path, 'r', 'utf8'), word_window=args.word_window),
                                       TASK.read_train_file(codecs.open(args.test_path, 'r', 'utf8'), word_window=args.word_window))

    sess = tf.Session()
    model = Model(TASK.scope, sess)

    model.train(train_data=train_data,
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
                evaluator=TASK.evaluator,
                eval_batch_size=args.eval_batch_size,
                print_freq=50,
                pre_trained_emb_path=args.pre_trained_emb_path,
                pre_trained_word_emb_path=args.pre_trained_word_emb_path,
                fix_word_emb=args.fix_word_emb,
                reserve_all_word_emb=args.reserve_all_word_emb,
                max_epoches=args.max_epoches)
    sess.close()
