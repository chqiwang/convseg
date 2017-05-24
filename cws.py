from __future__ import print_function
import os, codecs
from itertools import izip
from tagger import data_iterator


scope = 'CWS'


def process_train_sentence(sentence, bigram, word_window):
    sentence = sentence.strip()
    words = sentence.split()
    chars = []
    tags = []
    ret = []
    for w in words:
        chars.extend(list(w))
        if len(w) == 1:
            tags.append('S')
        else:
            tags.extend(['B'] + ['M'] * (len(w) - 2) + ['E'])
    ret.append(chars)
    if bigram:
        chars = ['', ''] + chars + ['', '']
        ret.append([a + b if a and b else '' for a, b in zip(chars[:-4], chars[1:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[1:-3], chars[2:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[2:-2], chars[3:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[3:-1], chars[4:])])
    elif word_window > 0:
        chars = ['', '', ''] + chars + ['', '', '']
        # single char
        if word_window >= 1:
            ret.append(chars[3:-3])
        if word_window >= 2:
            # bi chars
            ret.append([a + b if a and b else '' for a, b in zip(chars[2:], chars[3:-3])])
            ret.append([a + b if a and b else '' for a, b in zip(chars[3:-3], chars[4:])])
        if word_window >= 3:
            # tri chars
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[1:], chars[2:], chars[3:-3])])
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[2:], chars[3:-3], chars[4:])])
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[3:-3], chars[4:], chars[5:])])
        if word_window >= 4:
            # four chars
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[0:], chars[1:], chars[2:], chars[3:-3])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[1:], chars[2:], chars[3:-3], chars[4:])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[2:], chars[3:-3], chars[4:], chars[5:])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                             zip(chars[3:-3], chars[4:], chars[5:], chars[6:])])
    ret.append(tags)
    return ret


def process_raw_sentence(sentence, bigram, word_window):
    sentence = sentence.strip()
    chars = list(sentence)
    ret = [chars]
    if bigram:
        chars = ['', ''] + chars + ['', '']
        ret.append([a + b if a and b else '' for a, b in zip(chars[:-4], chars[1:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[1:-3], chars[2:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[2:-2], chars[3:])])
        ret.append([a + b if a and b else '' for a, b in zip(chars[3:-1], chars[4:])])
    elif word_window > 0:
        chars = ['', '', ''] + chars + ['', '', '']
        # single char
        if word_window >= 1:
            ret.append(chars[3:-3])
        if word_window >= 2:
            # bi chars
            ret.append([a + b if a and b else '' for a, b in zip(chars[2:], chars[3:-3])])
            ret.append([a + b if a and b else '' for a, b in zip(chars[3:-3], chars[4:])])
        if word_window >= 3:
            # tri chars
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[1:], chars[2:], chars[3:-3])])
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[2:], chars[3:-3], chars[4:])])
            ret.append(
                [a + b + c if a and b and c else '' for a, b, c in zip(chars[3:-3], chars[4:], chars[5:])])
        if word_window >= 4:
            # four chars
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[0:], chars[1:], chars[2:], chars[3:-3])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[1:], chars[2:], chars[3:-3], chars[4:])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                            zip(chars[2:], chars[3:-3], chars[4:], chars[5:])])
            ret.append([a + b + c + d if a and b and c and d else '' for a, b, c, d in
                             zip(chars[3:-3], chars[4:], chars[5:], chars[6:])])
    return ret


def read_train_file(fin, bigram=False, word_window=4):
    """
    Read training data.
    """
    data = []
    for l in fin:
        data.append(process_train_sentence(l, bigram, word_window))
    return zip(*data)


def read_raw_file(fin, batch_size, bigram=False, word_window=4):
    """
    Read raw data.
    """
    buffer = []
    max_buffer = 100000
    for i, l in enumerate(fin):
        buffer.append(process_raw_sentence(l, bigram, word_window))
        if i % max_buffer == 0 and i > 0:
            for b in data_iterator(zip(*buffer), batch_size, shuffle=False):
                yield b
            buffer = []
    if buffer:
        for b in data_iterator(zip(*buffer), batch_size, shuffle=False):
            yield b


def read_raw_file_all(fin, bigram=False, word_window=4):
    """
    Read raw data.
    """
    data = []
    for b in read_raw_file(fin, 1000, bigram, word_window):
        data.extend(zip(*b))
    return zip(*data)


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
                new_sen.append('  ')
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
    # Empty words file.
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