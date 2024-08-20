
from itertools import cycle
import random

import torch
from source.config.config import ModelGeneralConfig, TaskName
import torch.nn.functional as F
count = 0
def process_sentiment_batch(batch, objects_group: ModelGeneralConfig, args: dict):
    '''This function processes a batch of SST data. It takes as input a batch of data, a group of objects (model, optimizer, scheduler, etc.), 
    and the arguments. It returns the loss of the batch.'''
    device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
    model, optimizer = objects_group.model, objects_group.optimizer
    
    optimizer.zero_grad()
 


    b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
    b_ids, b_mask, b_labels = b_ids.to(device), b_mask.to(device), b_labels.to(device)

    logits = model.predict_sentiment(b_ids, b_mask)
  
    
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
    loss_value = loss.item()
    
    # objects_group.loss_sum += loss_value
    loss.backward()

    global count
    count = count + 1
    if count % 50 == 0:
        for name, param in model.named_parameters():
            with open('test.txt', 'a') as file:
                if param.grad is not None:
                    print(f"{name}:", file=file)   

    optimizer.step()
    
    return loss


def process_paraphrase_batch(batch, objects_group: ModelGeneralConfig, args: dict):
    '''This function processes a batch of paraphrase data. It takes as input a batch of data, 
    a group of objects (model, optimizer, scheduler, etc.), and the arguments. It returns the loss of the batch.'''
    device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
    model, optimizer = objects_group.model, objects_group.optimizer
    optimizer.zero_grad()

    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

    preds = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    loss = F.binary_cross_entropy_with_logits(preds.view(-1), b_labels.float(), reduction='sum') / args.batch_size
    loss_value = loss.item()


    # objects_group.loss_sum += loss_value
    
    loss.backward()
    optimizer.step()

    return loss


def process_similarity_batch(batch, objects_group: ModelGeneralConfig, args: dict):
    '''This function processes a batch of similarity data. It takes as input a batch of data,
    a group of objects (model, optimizer, scheduler, etc.), and the arguments. It returns the loss of the batch.'''
    device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
    model, optimizer = objects_group.model, objects_group.optimizer
    optimizer.zero_grad()

    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = b_ids_1.to(device), b_mask_1.to(device), b_ids_2.to(device), b_mask_2.to(device), b_labels.to(device)

    preds = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

    loss = F.mse_loss(preds.view(-1), b_labels.view(-1), reduction='sum') / args.batch_size

    loss_value = loss.item()

    # objects_group.loss_sum += loss_value
    loss.backward()
    optimizer.step()
    
    return loss


class Scheduler:
    '''A class to manage the learning rate scheduler.'''

    def __init__(self, dataloaders, reset=True):
        self.dataloaders = dataloaders
        self.names = list(dataloaders.keys())
        if reset: self.reset()

    def reset(self):
        self.sst_iter = iter(self.dataloaders[TaskName.SST])
        self.para_iter = iter(self.dataloaders[TaskName.PARAPHRASE])
        self.sts_iter = iter(self.dataloaders[TaskName.STS])
        self.steps = { TaskName.SST: 0, TaskName.PARAPHRASE: 0, TaskName.STS: 0}

    def get_SST_batch(self):
        try:
            return next(self.sst_iter)
        except StopIteration:
            self.sst_iter = cycle(self.dataloaders[TaskName.SST])
            return next(self.sst_iter)

    def get_Paraphrase_batch(self):
        try:
            return next(self.para_iter)
        except StopIteration:
            self.para_iter = cycle(self.dataloaders[TaskName.PARAPHRASE])
            return next(self.para_iter)

    def get_STS_batch(self):
        try:
            return next(self.sts_iter)
        except StopIteration:
            self.sts_iter = cycle(self.dataloaders[TaskName.STS])
            return next(self.sts_iter)

    def get_batch(self, name: str):
        if name == TaskName.SST: return self.get_SST_batch()
        elif name == TaskName.PARAPHRASE: return self.get_Paraphrase_batch()
        elif name == TaskName.STS: return self.get_STS_batch()
        raise ValueError(f"Unknown batch name: {name}")

    def process_named_batch(self, objects_group: ModelGeneralConfig, args: dict, name: str, apply_optimization: bool = True):
        '''Processes a batch of data from the given dataset, and updates the model accordingly.'''
        batch = self.get_batch(name)
        process_fn, gradient_accumulations = None, 0
        if name == TaskName.SST:
            process_fn = process_sentiment_batch
        elif name == TaskName.PARAPHRASE:
            process_fn = process_paraphrase_batch
        elif name == TaskName.STS:
            process_fn = process_similarity_batch
        else:
            raise ValueError(f"Unknown batch name: {name}")
        
        # Process the batch
        loss_of_batch = process_fn(batch, objects_group, args)

        # Update the model
        self.steps[name] += 1

        return loss_of_batch
    
class RandomScheduler(Scheduler):
    '''A scheduler that randomly chooses a batch to process.'''
    def __init__(self, dataloaders):
        super().__init__(dataloaders, reset=True)

    def process_one_batch(self, objects_group: ModelGeneralConfig, args: dict, name_specifier: str = None):
        
        if name_specifier is not None and name in self.names:
            name = name_specifier
        else:
            name = random.choice(self.names)

        return name, self.process_named_batch(objects_group, args, name)