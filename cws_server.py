# -*- coding:utf-8 -* 
from __future__ import print_function
from urllib import quote
import cPickle as pickle
import os
from itertools import izip
from argparse import ArgumentParser
import tornado.ioloop
import tornado.web
import tensorflow as tf

from tagger import build_input_graph, \
                   build_tagging_graph, data_to_ids, create_input, inference,\
                   INT_TYPE
from cws import create_output

page='''
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script>
function loadXMLDoc()
{
    var xmlhttp;
    if (window.XMLHttpRequest)
    {
        // IE7+, Firefox, Chrome, Opera, Safari
        xmlhttp=new XMLHttpRequest();
    }
    else
    {
        // IE6, IE5
        xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
    }
    xmlhttp.onreadystatechange=function()
    {
        if (xmlhttp.readyState==4 && xmlhttp.status==200)
        {
            document.getElementById("out").innerHTML=decodeURI(xmlhttp.responseText);
        }
    }

    var sentences=document.getElementById("in").value
    var req="sentences="+sentences
    req=encodeURI(req)
    xmlhttp.open("POST","/cws",true);
    xmlhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xmlhttp.send(req);
}
</script>
</head>
<body>
<div>
<textarea id="in" style="height:400px;width:500px;font-size:18px"></textarea>
<button onclick="loadXMLDoc()" style="height:25px;width:80px;font-size:18px">分词</button>
<textarea id="out" style="height:400px;width:500px;font-size:18px"></textarea>
</div>
<br>
<br>
This is a Chinese word segmentation (CWS) demo provided by Chunqi Wang (chqiwang@126.com)

</body>
</html>
'''


class Tagger(object):
    def __init__(self, sess, model_dir, scope):
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

        self.scope = scope
        self.sess = sess
        self.seq_ids_pl = seq_ids_pl
        self.seq_lengths_pl = seq_lengths_pl
        self.seq_other_ids_pls = seq_other_ids_pls
        self.is_train_pl = is_train_pl
        self.scores_op = scores_op
        self.item2id = item2id
        self.word2id = word2id
        self.id2tag = id2tag
        self.parameters = parameters

    def tag(self, sentences):
        data = self.preprocess(sentences)
        batch = data_to_ids(data, [self.item2id] + [self.word2id] * self.parameters['word_window_size'])
        batch = create_input(batch)
        seq_ids, seq_other_ids_list, seq_lengths = batch[0], batch[1: -1], batch[-1]
        feed_dict = {self.seq_ids_pl: seq_ids.astype(INT_TYPE),
             self.seq_lengths_pl: seq_lengths.astype(INT_TYPE),
             self.is_train_pl: False}
        for pl, v in zip(self.seq_other_ids_pls, seq_other_ids_list):
            feed_dict[pl] = v.astype(INT_TYPE)
        scores = self.sess.run(self.scores_op, feed_dict)
        if self.parameters['use_crf']:
            with tf.variable_scope(self.scope, reuse=True):
                stag_ids = inference(scores, seq_lengths, tf.get_variable('transitions').eval(session=self.sess))
        else:
            stag_ids = inference(scores, seq_lengths)
        output = []
        for seq, stag_id, length in izip(data[0], stag_ids, seq_lengths):
            output.append((seq, [self.id2tag[t] for t in stag_id[:length]]))
        return create_output(*zip(*output))

    def preprocess(self, sentences):
        sentences = sentences.split('\n')
        data = ([], [], [], [], [], [], [], [], [], [], [], [])
        for l in sentences:
            l = l.strip()
            chars = list(l)
            data[0].append(chars)
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
        return data


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(page)


class CWSHandler(tornado.web.RequestHandler):
    def initialize(self, tagger):
        self.tagger = tagger

    def post(self):
        sentences = self.get_argument('sentences')
        segs = self.tagger.tag(sentences)
        self.write(quote('\n'.join(segs).encode('utf8')))


def make_app(model_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    tagger = Tagger(sess=sess, model_dir=model_dir, scope='CWS')
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/cws", CWSHandler, {'tagger': tagger})
    ])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--port', dest='port', type=int, default=8888)
    args = parser.parse_args()
    
    app = make_app(args.model_dir)
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
