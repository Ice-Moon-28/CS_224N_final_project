'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from source.model.bert import BertModel
from source.config.config import ModelGeneralConfig, TaskName
from source.train.optimizer import AdamW

from source.dataset.datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from source.train.evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask, models_eval_multitask
from source.train.scheduler import RandomScheduler, Scheduler, process_paraphrase_batch, process_sentiment_batch, process_similarity_batch
from source.model.tokenizer import BertTokenizer


TQDM_DISABLE=False


# Fix the random seed.


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5



class ClassifierLayer(nn.Module):
    def __init__(self, config, objective_classes, taskName = None):
        super().__init__()
        if taskName == TaskName.SST:
            dropout_capacity = config.hidden_dropout_prob_sst
        elif taskName == TaskName.PARAPHRASE:
            dropout_capacity = config.hidden_dropout_prob_para
        elif taskName == TaskName.STS:
            dropout_capacity = config.hidden_dropout_prob_sts
        else:
            dropout_capacity = config.hidden_dropout_prob

        self.linear_layers = config.n_hidden_layers
        self.linear = nn.ModuleList([nn.Linear(config.hidden_size, config.bert_hidden_size) for _ in range(config.n_hidden_layers - 1)])
        
        self.dropout = nn.ModuleList([nn.Dropout(dropout_capacity) for _ in range(config.n_hidden_layers)])
        self.final_linear = nn.Linear(config.bert_hidden_size, objective_classes)

    def forward(self, hidden_states):
        for i in range(self.linear_layers - 1):
            hidden_states = self.linear[i](hidden_states)
            hidden_states = self.dropout[i](hidden_states)
            hidden_states = F.relu(hidden_states)
        
        hidden_states = self.final_linear(hidden_states)
        hidden_states = self.dropout[-1](hidden_states)

        return hidden_states
        

class MultitaskBERTClassifier(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True

        self.sentiment_classifier = ClassifierLayer(config, N_SENTIMENT_CLASSES, TaskName.SST)

        self.sentiment_classifier.requires_grad_ = True

        self.paraphrase_classifier = ClassifierLayer(config, 1, TaskName.PARAPHRASE)

        self.paraphrase_classifier.requires_grad_ = True

        self.similarity_classifier = ClassifierLayer(config, 1, TaskName.STS)

        self.similarity_classifier.requires_grad_ = True

    def predict_sentiment(self, input_ids, attention_mask, task_id = None): 
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        embeddings = self.forward(input_ids, attention_mask)

        logits = self.sentiment_classifier(embeddings)

        return logits

    def forward(self, input_ids, attention_mask, task_id = None):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        
   
        bert_output = self.bert(input_ids, attention_mask)


        # Step 2: Get the [CLS] token embeddings
        cls_embeddings = bert_output['pooler_output']
        return cls_embeddings


    def get_similarity_paraphrase_embeddings(self, input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2, task_id = None):
        '''Given a batch of pairs of sentences, get the BERT embeddings.'''
        # Step 0: Get [SEP] token ids
        sep_token_id = torch.tensor([self.tokenizer.sep_token_id], dtype=torch.long, device=input_ids_1.device)
        batch_sep_token_id = sep_token_id.repeat(input_ids_1.shape[0], 1)

        # Step 1: Concatenate the two sentences in: sent1 [SEP] sent2 [SEP]
        input_id = torch.cat((input_ids_1, batch_sep_token_id, input_ids_2, batch_sep_token_id), dim=1)
        attention_mask = torch.cat((attention_mask_1, torch.ones_like(batch_sep_token_id), attention_mask_2, torch.ones_like(batch_sep_token_id)), dim=1)

        # Step 2: Get the BERT embeddings
        x = self.forward(input_id, attention_mask, task_id=task_id)

        return x

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           task_id = None
                           ):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        # Step 1: Get the BERT embeddings
        embeddings = self.get_similarity_paraphrase_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id=1)
        
        similarity = self.paraphrase_classifier(embeddings)
        return similarity

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2,
                           task_id = None):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # Step 1 : Get the BERT embeddings
        embeddings = self.get_similarity_paraphrase_embeddings(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, task_id=2)

        similarity = self.similarity_classifier(embeddings)

        return similarity








    