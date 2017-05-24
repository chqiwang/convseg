# -*- coding:utf-8 -* 
from __future__ import print_function
import StringIO
from argparse import ArgumentParser
import tornado.ioloop
import tornado.web
import tensorflow as tf

from tagger import Model


def load_template():
    page = '''
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
        xmlhttp.open("POST","/%s",true);
        xmlhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        xmlhttp.send(req);
    }
    </script>
    </head>
    <body>
    <div>
    <textarea id="in" style="height:400px;width:500px;font-size:18px"></textarea>
    <button onclick="loadXMLDoc()" style="height:25px;width:80px;font-size:18px">%s</button>
    <textarea id="out" style="height:400px;width:500px;font-size:18px"></textarea>
    </div>
    <br>
    <br>
    This is a %s demo provided by Chunqi Wang (chqiwang@126.com).
    </body>
    </html>
    ''' % (TASK.scope, TASK.scope, TASK.scope)
    return page


class Tagger(object):
    def __init__(self, sess, model_dir, scope, batch_size):
        self.model = Model(scope, sess)
        self.batch_size = batch_size
        self.model.load_model(model_dir)

    def tag(self, sentences):
        sentences = self.preprocess(sentences)
        sf = StringIO.StringIO()
        sf.write(sentences)
        sf.seek(0)
        data = TASK.read_raw_file_all(sf)
        output = self.model.tag_all(data, self.batch_size)
        return TASK.create_output(*output)

    def preprocess(self, sentences):
        return sentences


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write(load_template())


class TaskHandler(tornado.web.RequestHandler):
    def initialize(self, tagger):
        self.tagger = tagger

    def post(self):
        sentences = self.get_argument('sentences')
        segs = self.tagger.tag(sentences)
        self.write('\n'.join(segs))


def make_app(model_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)
    tagger = Tagger(sess=sess, model_dir=model_dir, scope=TASK.scope, batch_size=200)
    return tornado.web.Application([
        (r"/", MainHandler),
        (r"/%s" % TASK.scope, TaskHandler, {'tagger': tagger})
    ])


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--task', dest='task')
    parser.add_argument('--model_dir', dest='model_dir')
    parser.add_argument('--port', dest='port', type=int, default=8888)
    args = parser.parse_args()

    TASK = __import__(args.task)

    app = make_app(args.model_dir)
    app.listen(args.port)
    tornado.ioloop.IOLoop.current().start()
