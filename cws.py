from __future__ import print_function
import codecs
import time
from argparse import ArgumentParser

import tensorflow as tf
from tagger import tag
from train_cws import create_output


def read_raw_text(path, batch_size, bigram=False, word=True):
    """
    Read raw data.
    """
    if bigram:
        data = ([], [], [], [], [])
    elif word:
        data = ([], [], [], [], [], [], [], [], [], [], [])
    else:
        data = ([], )

    for i, l in enumerate(codecs.open(path, 'r', 'utf8')):
        chars = list(l.strip())
        data[0].append(chars)
        if bigram:
            chars = ['', ''] + chars + ['', '']
            data[1].append([a + b if a and b else '' for a, b in zip(chars[:-4], chars[1:])])
            data[2].append([a + b if a and b else '' for a, b in zip(chars[1:-3], chars[2:])])
            data[3].append([a + b if a and b else '' for a, b in zip(chars[2:-2], chars[3:])])
            data[4].append([a + b if a and b else '' for a, b in zip(chars[3:-1], chars[4:])])
        elif word:
            chars = ['', '', ''] + chars + ['', '', '']
            # single char
            data[1].append(chars[3:-3])
            # bi chars
            data[2].append([a + b if a and b else '' for a, b in zip(chars[2:], chars[3:-3])])
            data[3].append([a + b if a and b else '' for a, b in zip(chars[3:-3], chars[4:])])
            # tri chars
            data[4].append([a + b + c if a and b and c else '' for a, b, c in zip(chars[1:], chars[2:], chars[3:-3])])
            data[5].append([a + b + c if a and b and c else '' for a, b, c in zip(chars[2:], chars[3:-3], chars[4:])])
            data[6].append([a + b + c if a and b and c else '' for a, b, c in zip(chars[3:-3], chars[4:], chars[5:])])
            # four chars
            data[7].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[0:], chars[1:], chars[2:], chars[3:-3])])
            data[8].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[1:], chars[2:], chars[3:-3], chars[4:])])
            data[9].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[2:], chars[3:-3], chars[4:], chars[5:])])
            data[10].append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                             zip(chars[3:-3], chars[4:], chars[5:], chars[6:])])
        if i % batch_size == 0 and i > 0:
            yield data
            if bigram:
                data = ([], [], [], [], [])
            elif word:
                data = ([], [], [], [], [], [], [], [], [], [], [])
            else:
                data = ([],)
    if data:
        yield data


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input', dest='input')
    parser.add_argument('--output', dest='output')
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--batch_size', dest='batch_size', type=int)

    args = parser.parse_args()
    data_iter = read_raw_text(args.input, args.batch_size)

    fout = codecs.open(args.output, 'w', 'utf8')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.allow_soft_placement = True
    config.log_device_placement = True

    with tf.Session(config=config) as sess:
        start = time.time()
        count = 0
        for seqs, stags in tag(sess, data_iter, args.model_dir, 'CWS'):
            for l in create_output(seqs, stags):
                count += 1
                print(l, file=fout)
            print('Tagged %d lines in %d seconds.' % (count, time.time() - start))
        end = time.time()

    fout.close()
    print('Finished.')
