import argparse
import os.path

import numpy
import torch
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from evaluation.assin.Assin2Processor import preprocess_assin2
from evaluation.assin.AssinDataset import AssinDataset
from tokenization.tokenizer_loader import TokenizerLoader
from utils.model_selector import load_from_checkpoint
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Script that loads a model's finetune checkpoint and evaluates it for ASSIN")
    parser.add_argument('-run_name', dest='runName', help='Pretrain run to choose assin ft checkpoint from')
    parser.add_argument('-base_model', dest='baseModel', help='Base Model - GPTNEO-350, GPTNEO-1.3B, etc')
    parser.add_argument('-run_checkpoint', dest='runCheckpoint',
                        help='Pretrain run checkpoint to choose assin ft checkpoint from')
    parser.add_argument('-assin_task', dest='assinTask', default="assin_entailment",
                        help='Assin task to eval. Entailment, Similarity or BOTH. Last option'
                             'requires two checkpoints with in same name, one for each task.')
    parser.add_argument('-assin_ver', dest='assinVersion', default=2, type=int, help='Ass')
    parser.add_argument('-targetCheckpoint', dest='targetCheckpoint',
                        help='Target assin ft checkpoint to eval for the given task')
    args = parser.parse_args()

    print(args)

    # Load tokenizer
    tokenizerLoader = TokenizerLoader("gptuga-tk-512")
    loaded_tokenizer = tokenizerLoader.loadTokenizer(128, args.baseModel)

    # Load assin - version 1 comes from huggingface, while the second needs to
    # be locally stored and loaded, since the huggingface version is not correct
    if args.assinVersion == "1":
        # Load assin1 https://huggingface.co/datasets/assin/viewer/full/train
        data = load_dataset("assin", name="ptpt")['test']
    else:
        # Load assin2
        data = preprocess_assin2("assin2_pickled")['test']

    testData = AssinDataset(loaded_tokenizer=loaded_tokenizer,
                                batchSize=32,  # assin_params['batchSize'],
                                data=data,
                                seq_len=128,
                                categoric=False,
                                multitask=True)

    data_collator = DataCollatorWithPadding(tokenizer=loaded_tokenizer, max_length=128)

    # Train dataloader
    test_dataloader = DataLoader(
        testData, collate_fn=data_collator, batch_size=64,
    )

    # The following is just logic to load the target checkpoint
    # Feel free to change this according to your own folder organization
    model_dir = os.path.join("/data/rv.lopes/models", args.runName)  # ../model
    target_model_checkpoints_dir = os.path.join(model_dir, "checkpoints")  # ../model/checkpoints
    target_chk = os.path.join(target_model_checkpoints_dir, args.runCheckpoint)  # ../model/checkpoints/checkpoint-1000

    metrics_checkpoints = ['checkpoint-best-f1', 'checkpoint-best-pearson']
    for metric_chk in metrics_checkpoints:
        print("Evaluating ASSIN for checkpoitn:", metric_chk)
        if args.assinVersion == 2:
            target_assin_ft_check = target_chk + "/finetune/assin-2/ftcheckpoints/" + metric_chk
        else:
            target_assin_ft_check = target_chk + "/finetune/assin-1/ftcheckpoints/" + metric_chk

        model, config = load_from_checkpoint(baseModel=args.baseModel, ft_task=args.assinTask,
                                             targetCheck=target_assin_ft_check)

        gold_similarity = []
        gold_entailment = []
        sys_similarities = numpy.array([])
        sys_entailments = numpy.array([])
        docs = []
        model.to("cuda")
        model.eval()
        for batch in test_dataloader:
            batch.to("cuda")
            with torch.no_grad():
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                labels=batch['labels'])

            # Unpack label pairs: [RTE, STS] -> [RTE] [STS]
            aux_labels = [[], []]
            for label_pair in batch['labels']:
                aux_labels[0] += [label_pair[0].cpu().item()]
                aux_labels[1] += [label_pair[1].cpu().item()]

            # Gold predictions for batch
            gold_entailment += aux_labels[0]
            gold_similarity += aux_labels[1]

            # RTE PREDS
            rte_preds = outputs.logits[0].cpu().argmax(dim=1).numpy()
            sys_entailments = numpy.concatenate((sys_entailments, rte_preds))

            # STS PREDS
            sts_preds = outputs.logits[1].detach().cpu().squeeze()
            sys_similarities = numpy.concatenate((sys_similarities, sts_preds))


        macro_f1 = f1_score(gold_entailment, sys_entailments, average='macro',
                            labels=list(gold_entailment))
        accuracy = (gold_entailment == sys_entailments).sum() / len(gold_entailment)
        pearson = pearsonr(gold_similarity, sys_similarities)[0]
        absolute_diff = gold_similarity - sys_similarities
        mse = (absolute_diff ** 2).mean()

        print()
        print('RTE evaluation')
        print('Accuracy\tMacro F1')
        print('--------\t--------')
        print('{:8.2%}\t{:8.3f}'.format(accuracy, macro_f1))

        print()
        print('Similarity evaluation')
        print('Pearson\t\tMean Squared Error')
        print('-------\t\t------------------')
        print('{:7.3f}\t\t{:18.2f}'.format(pearson, mse))
