# CONV-SEG
Convolutional neural network for Chinese word segmentation (CWS).

## Dependencies
* Python 2.7
* [Tensorflow 1.0](https://www.tensorflow.org/)

It is better to use a nvidia GPU to accelerate the training procedure.

## Data
Downlaod `data.zip` from <https://drive.google.com/open?id=0B-f0oKMQIe6sQVNxeE9JeUJfQ0k>. Extract `data.zip` to this directory. So the file tree would be:

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
	
More arguments can be set in `train_cws.py`.