# CS 224N Default Final Project - Multitask BERT

This is the default final project for the Stanford CS 224N class. Please refer to the project handout on the course website for detailed instructions and an overview of the codebase.

This project comprises two parts. In the first part, you will implement some important components of the BERT model to better understand its architecture. 
In the second part, you will use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection, and semantic similarity. You will implement extensions to improve your model's performance on the three downstream tasks.

In broad strokes, Part 1 of this project targets:
* bert.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

And Part 2 targets:
* multitask_classifier.py: Missing code blocks.
* datasets.py: Possibly useful functions/classes for extensions.
* evaluation.py: Possibly useful functions/classes for extensions.

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

Then, run `python3 main.py` to train the model, there are a lots of config to specific....

Just specific two important choices:

<ul>
<li>ensemble: to enable ensemble process in evaluation, training process</li>
<li>fine-tune-mode: to choose from two choices: just train the last classifier layer or fine-tune the whole model</li>
<li>use_cuda: to train model with cuda</li>
<li>use_gpu: to train model with gpu</li>
</ul>

## Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).