# CONV-SEG
Convolutional neural network for Chinese word segmentation (CWS).

## Author
The code is written and maintained by Chunqi Wang. If you have any problem in using this code, please email to <chqiwang@126.com>.

## Dependencies
* [Python 2.7](https://www.python.org/)
* [Tensorflow 1.0](https://www.tensorflow.org/)

It is better to use a nvidia GPU to accelerate the training procedure.

## Data
Downlaod `data.zip` from [here](https://drive.google.com/open?id=0B-f0oKMQIe6sQVNxeE9JeUJfQ0k) (Note that the SIGHAN datasets should only be used for research purposes). Extract `data.zip` to this directory. So the file tree would be:

	convseg
	|	data
	|	|	datasets
	|	|	|	sighan2005-pku
	|	|	|	|	train.txt
	|	|	|	|	dev.txt
	|	|	|	|	test.txt
	|	|	|	sighan2005-msr
	|	|	|	|	train.txt
	|	|	|	|	dev.txt
	|	|	|	|	test.txt
	|	|	embeddings
	|	|	|	news_tensite.w2v200
	|	|	|	news_tensite.pku.words.w2v50
	|	|	|	news_tensite.msr.words.w2v50
	|	tagger.py
	|	train_cws.py
	|	train_cws.sh
	|	train_cws_wemb.sh
	|	score.perl
	|	README.md

## How to use
First, give execute permission to scripts:

	chmod +x train_cws.sh train_cws_wemb.sh

Train a preliminary model (CONV-SEG):

	./train_cws.sh WHICH_DATASET WHICH_GPU
	
Train a model with word embeddings (WE-CONV-SEG):

	./train_cws_wemb.sh WHICH_DATASET WHICH_GPU
	
We have two optional datasets: `pku` and `msr`. If you run the program in CPU environment, just leave the second argument empty.

For example, if you want to train the model CONV-SEG on the pku dataset and on gpu0, you should run:

	./train_cws.sh pku 0
	
More arguments can be set in `train.py`.

## Test Score
| Model | PKU(dev) | PKU(test) | MSR(dev) | MSR(test) |
|:------|:---------|:----------|:---------|:----------|
| CONV-SEG | 96.8 | 95.7 | 97.2 | 97.3	|
| WE-CONV-SEG | 97.5 |	96.5	| 98.1 |	98.0 |

## Demo
Browse <http://ubuntu23447.cloudapp.net:8888> for a demo.