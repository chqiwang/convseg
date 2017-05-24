from __future__ import print_function
import codecs
import time
from argparse import ArgumentParser

import tensorflow as tf
from tagger import Model


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--task', dest='task')
    parser.add_argument('--input', dest='input')
    parser.add_argument('--output', dest='output')
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--batch_size', dest='batch_size', type=int)

    args = parser.parse_args()
    TASK = __import__(args.task)

    data_iter = TASK.read_raw_file(codecs.open(args.input, 'r', 'utf8'), args.batch_size)

    fout = codecs.open(args.output, 'w', 'utf8')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.allow_soft_placement = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:
        model = Model(TASK.scope, sess)
        model.load_model(args.model_dir)
        start = time.time()
        count = 0
        for seqs, stags in model.tag(data_iter):
            for l in TASK.create_output(seqs, stags):
                count += 1
                print(l, file=fout)
            print('Tagged %d lines in %d seconds.' % (count, time.time() - start))
        end = time.time()

    fout.close()
    print('Finished.')
