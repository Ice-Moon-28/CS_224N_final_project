
import random
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from source.config.config import ModelGeneralConfig, TaskName
from torch.utils.data import Dataset, DataLoader
from source.train.evaluation import TQDM_DISABLE, model_eval_multitask, models_eval_multitask
from source.dataset.datasets import SentenceClassificationDataset, SentenceClassificationTestDataset, SentencePairDataset, SentencePairTestDataset, load_multitask_data
from source.model.multitask_classifier import MultitaskBERTClassifier
from source.train.optimizer import AdamW
from source.train.scheduler import RandomScheduler, process_paraphrase_batch, process_sentiment_batch, process_similarity_batch

import os

from source.util.utils import computer_parameter_of_model, model_reload_from_file


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_full_model(args, train_dataloaders, dev_dataloaders, object_groups: ModelGeneralConfig):

    device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
    scheduler = RandomScheduler(train_dataloaders)
    model, optimizer, config = object_groups.model, object_groups.optimizer, object_groups.modelConfig

    sst_train_dataloader, para_train_dataloader, sts_train_dataloader = train_dataloaders[TaskName.SST], train_dataloaders[TaskName.PARAPHRASE], train_dataloaders[TaskName.STS]

    sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader = dev_dataloaders[TaskName.SST], dev_dataloaders[TaskName.PARAPHRASE], dev_dataloaders[TaskName.STS]

    best_dev_acc = 0
    num_batches_per_epoch = config.num_batch_per_epoch
    n_batches = 0

    total_num_batches = { TaskName.SST: 0, TaskName.PARAPHRASE: 0, TaskName.STS: 0}
    train_loss = { TaskName.SST: 0, TaskName.PARAPHRASE: 0, TaskName.STS: 0}
    num_batches = { TaskName.SST: 0, TaskName.PARAPHRASE: 0, TaskName.STS: 0}

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(range(num_batches_per_epoch), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            task, loss = scheduler.process_one_batch(objects_group=object_groups, args=args)
            n_batches += 1
            train_loss[task] += loss.item()
            num_batches[task] += 1
            total_num_batches[task] += 1

        for task in [TaskName.SST, TaskName.PARAPHRASE, TaskName.STS]:
            if num_batches[task] > 0:
                train_loss[task] = train_loss[task] / (num_batches[task])
            else:
                train_loss[task] = 0
        
        train_avg_loss = sum([loss for loss in train_loss.values()]) / len(train_loss)

        writeLogIntoFileAndPrint(f'{train_loss} ======= train loss =========')

        # (sst_train_acc, _, _,
        # para_train_acc, _, _,
        # sts_train_acc, _, _) = model_eval_multitask(
        #     sentiment_dataloader=sst_train_dataloader,
        #     paraphrase_dataloader=para_train_dataloader,
        #     sts_dataloader=sts_train_dataloader,
        #     model=model,
        #     device=device,
        #     enable_sst=True,
        #     enable_para=True,
        #     enable_sts=True,
        # )

        # writeLogIntoFileAndPrint(f'Epoch {epoch}: train sst acc :: {sst_train_acc :.3f}, train para acc :: {para_train_acc :.3f}, train sts acc :: {sts_train_acc :.3f}')

        (sst_dev_acc, _, _,
        para_dev_acc, _, _,
        sts_dev_acc, _, _) = model_eval_multitask(
            sentiment_dataloader=sst_dev_dataloader,
            paraphrase_dataloader=para_dev_dataloader,
            sts_dataloader=sts_dev_dataloader,
            model=model,
            device=device,
        )

        writeLogIntoFileAndPrint(f"Epoch {epoch}: dev sst acc :: {sst_dev_acc :.3f}, dev para acc :: {para_dev_acc :.3f}, dev sts acc :: {sts_dev_acc :.3f}")

        dev_acc = (sst_dev_acc + para_dev_acc + sts_dev_acc) / 3

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        writeLogIntoFileAndPrint(f"Epoch {epoch}: train loss :: {train_avg_loss :.3f}, dev acc :: {dev_acc :.3f}")

def train_last_layers(args, train_dataloaders, dev_dataloaders, object_groups: ModelGeneralConfig):

    for task in [TaskName.SST, TaskName.PARAPHRASE, TaskName.STS]:
        print(f'Training {task}')
       
        train_simple_task(
            args=args,
            train_dataloaders=train_dataloaders,
            dev_dataloaders=dev_dataloaders,
            objects_group=object_groups,
            taskName=task,
        )


def train_simple_task(args, train_dataloaders, dev_dataloaders, objects_group: ModelGeneralConfig , taskName):
    device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')

    model, modelConfig = objects_group.model, objects_group.modelConfig
    best_dev_acc = 0
    if taskName == TaskName.SST:
        epochs = args.epochs_sst
    elif taskName == TaskName.PARAPHRASE:
        epochs = args.epochs_paraphrase
    elif taskName == TaskName.STS:
        epochs = args.epochs_sts
    else:
        epochs = args.epochs
    

    dataloader = train_dataloaders[taskName]
    for epoch in range(epochs):
        model.train()
        n_batches = 0
        train_loss = 0
        optimizer = AdamW(model.parameters(), lr=args.lr)
        objects_group = ModelGeneralConfig(model=model, optimizer=optimizer, scaler=None)
    
        for batch in tqdm(dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            if taskName == TaskName.SST:
                loss = process_sentiment_batch(batch, objects_group, args)
            elif taskName == TaskName.PARAPHRASE:
                loss = process_paraphrase_batch(batch, objects_group, args)
            elif taskName == TaskName.STS:
                loss = process_similarity_batch(batch, objects_group, args)
            n_batches += 1
            train_loss += loss.item()

        train_loss = train_loss / (n_batches)        

        (sst_train_acc, _, _,
        para_train_acc, _, _,
        sts_train_acc, _, _) = model_eval_multitask(
            sentiment_dataloader=train_dataloaders[TaskName.SST],
            paraphrase_dataloader=train_dataloaders[TaskName.PARAPHRASE],
            sts_dataloader=train_dataloaders[TaskName.STS],
            model=model,
            device=device,
            enable_sst=taskName == TaskName.SST,
            enable_para=taskName == TaskName.PARAPHRASE,
            enable_sts=taskName == TaskName.STS,
        )

        print(f'Epoch {epoch}: train sst acc :: {sst_train_acc :.3f}, train para acc :: {para_train_acc :.3f}, train sts acc :: {sts_train_acc :.3f}')


        (sst_dev_acc, _, _,
        para_dev_acc, _, _,
        sts_dev_acc, _, _) = model_eval_multitask(
            sentiment_dataloader=dev_dataloaders[TaskName.SST],
            paraphrase_dataloader=dev_dataloaders[TaskName.PARAPHRASE],
            sts_dataloader=dev_dataloaders[TaskName.STS],
            model=model,
            device=device,
            enable_sst=taskName == TaskName.SST,
            enable_para=taskName == TaskName.PARAPHRASE,
            enable_sts=True,
            # enable_sts=taskName == TaskName.STS,
        )

        print(f"Epoch {epoch}: dev sst acc :: {sst_dev_acc :.3f}, dev para acc :: {para_dev_acc :.3f}, dev sts acc :: {sts_dev_acc :.3f}")
        if taskName == TaskName.SST:
            dev_acc = sst_dev_acc
        elif taskName == TaskName.PARAPHRASE:
            dev_acc = para_dev_acc
        elif taskName == TaskName.STS:
            dev_acc = sts_dev_acc

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, modelConfig, args.filepath)
        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")

    pass

def ensemble_data(dataList):
    print(len(dataList), type(dataList))
    return [dataList[random.randint(0, len(dataList) - 1)] for _ in range(len(dataList))]

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''

    device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    if args.ensemble:
        sst_train_data = ensemble_data(sst_train_data)
        sst_dev_data = ensemble_data(sst_dev_data)
        
        para_train_data = ensemble_data(para_train_data)
        para_dev_data = ensemble_data(para_dev_data)

        sts_train_data = ensemble_data(sts_train_data)
        sts_dev_data = ensemble_data(sts_dev_data)

     
    # SST: Sentiment classification
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    # Para: Paraphrase detection
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size_para,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size_para,
                                    collate_fn=para_dev_data.collate_fn)


    # STS: Semantic textual similarity
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size_sts,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size_sts,
                                    collate_fn=sts_dev_data.collate_fn)


    # Init model.
    config = {
                'hidden_dropout_prob': args.hidden_dropout_prob,
                'hidden_dropout_prob_sst': args.hidden_dropout_prob_sst,
                'hidden_dropout_prob_para': args.hidden_dropout_prob_para,
                'hidden_dropout_prob_sts': args.hidden_dropout_prob_sts,
                'num_labels': num_labels,
                'hidden_size': 768,
                'data_dir': '.',
                'fine_tune_mode': args.fine_tune_mode,
                'n_hidden_layers': 2,
                'bert_hidden_size': 512,
                'num_batch_per_epoch': args.num_batch_per_epoch,
            }

    config = SimpleNamespace(**config)

    model = MultitaskBERTClassifier(config)
    model = model.to(device)
    if args.recover:
        model_reload_from_file(model, args.filepath)
    train_dataloaders = { TaskName.SST: sst_train_dataloader, TaskName.PARAPHRASE: para_train_dataloader, TaskName.STS: sts_train_dataloader }
    dev_dataloaders = { TaskName.SST: sst_dev_dataloader, TaskName.PARAPHRASE: para_dev_dataloader, TaskName.STS: sts_dev_dataloader }


    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)


    modelGeneralConfig = ModelGeneralConfig(model=model, optimizer=optimizer, scaler=None, modelConfig=config)
    
    if args.fine_tune_mode == 'last-linear-layer':
        train_last_layers(
            args=args,
            train_dataloaders=train_dataloaders,
            dev_dataloaders=dev_dataloaders,
            object_groups=modelGeneralConfig,
        )
    else:
        train_full_model(
            args=args,
            train_dataloaders=train_dataloaders,
            dev_dataloaders=dev_dataloaders,
            object_groups=modelGeneralConfig,
        )


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERTClassifier(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")

def test_multitask_ensemble(args, filepaths):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():

        device = torch.device('mps') if args.use_gpu else torch.device('cpu') if not args.use_cuda else torch.device('cuda')
        
        models = [
            MultitaskBERTClassifier(torch.load(filepath)['model_config']).to(device) for filepath in filepaths
        ]

        #### 110M 的 parameter
        print(computer_parameter_of_model(model=models[0]), '== parameters of model ==')
        
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = models_eval_multitask(
                                                                    sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader,
                                                                    models,
                                                                    device,
                                                                    enable_para=True,
                                                                    enable_sts=True,
                                                                    enable_sst=True,
                                                                    )

        # test_sst_y_pred, \
        #     test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
        #         model_eval_test_multitask(sst_test_dataloader,
        #                                   para_test_dataloader,
        #                                   sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        # with open(args.sst_test_out, "w+") as f:
        #     f.write(f"id \t Predicted_Sentiment \n")
        #     for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
        #         f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        # with open(args.para_test_out, "w+") as f:
        #     f.write(f"id \t Predicted_Is_Paraphrase \n")
        #     for p, s in zip(test_para_sent_ids, test_para_y_pred):
        #         f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        # with open(args.sts_test_out, "w+") as f:
        #     f.write(f"id \t Predicted_Similiary \n")
        #     for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
        #         f.write(f"{p} , {s} \n")